// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_HODGELAPLACIAN_H
#define TESSERA_COBORDISM_HODGELAPLACIAN_H

#include <complex>

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cobordism/Cochain.h"
#include "cobordism/Spectrum.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # HodgeLaplacian
///
/// The Hodge Laplacian on a `Spacetime`, degree-parameterized by `int k`.
///
/// Reference: Lim, "Hodge Laplacians on graphs", arXiv:1507.05379; Horak and
/// Jost, "Spectra of combinatorial Laplace operators on simplicial complexes",
/// arXiv:1105.2712.
///
/// ## Metric source
///
/// The operator's metric is chosen by `MetricSource`. The default is the
/// chain-level Whitney pencil (`WhitneyPencil`): the metric weights
/// \f$ W_k = M_k^{-1} \f$ of the sparse Whitney mass matrices of the complex
/// squared edge lengths, dressed by the connection \f$ U \f$ read from the
/// edge phases, so that `laplacian(k)` is the covariant one-particle operator
/// \f$ h_k(s,U) \f$ written on geometric images and moves with \f$ U \f$.
/// Every read of the pipeline is taken on this operator; there is no separate
/// untwisted read. Its spectrum is invariant under the gauge similarity
/// \f$ \rho_k(g)^{-1} h_k(s,U)\rho_k(g) \f$, so the effective homology and
/// every spectral gate built on it are gauge-invariant, and the action's
/// Hodge-entropy term (`spectralEntropy`) depends on \f$ U \f$ through its
/// holonomy, which is what makes the connection dynamical. The diagonal
/// per-simplex weights described next (`DiagonalWeights`) remain available by
/// name; their operator does not see \f$ U \f$.
///
/// ## Definition under diagonal weights
///
/// From the oriented integer boundary maps \f$ \partial_k \f$ (`ChainComplex`),
/// the diagonal metric weight \f$ W_k \f$ on \f$ k \f$-chains (`weights`, from
/// `Simplex::volume`; \f$ W_0 = I \f$) and the weighted adjoint
/// \f$ \partial_k^{*} = W_k^{-1}\partial_k^{\dagger}W_{k-1} \f$:
/// \f[ L_k = \partial_{k+1}\partial_{k+1}^{*} + \partial_k^{*}\partial_k
///         = \partial_{k+1}W_{k+1}^{-1}\partial_{k+1}^{\dagger}W_k
///           + W_k^{-1}\partial_k^{\dagger}W_{k-1}\partial_k \f]
/// for every \f$ k \geq 0 \f$. At \f$ k = 0 \f$ the second term drops and
/// \f$ L_0 = \partial_1 W_1^{-1}\partial_1^{\dagger} \f$ is the graph
/// Laplacian with off-diagonal \f$ -1/W_1(e) \f$ on the 1-cell
/// \f$ e = (u,v) \f$ and diagonal \f$ \sum_{e \ni u} 1/W_1(e) \f$; its row
/// sums vanish at any weights, so \f$ \dim\ker L_0 = b_0 \f$. At *positive*
/// weights the discrete Hodge theorem \f$ \ker L_k \cong H_k \f$ holds at
/// every degree. `metric = false` selects unit weights, the combinatorial
/// \f$ \partial_{k+1}\partial_{k+1}^{\dagger} + \partial_k^{\dagger}\partial_k \f$.
/// A negative \f$ k \f$ throws; \f$ k \f$ beyond the top dimension yields
/// empty results.
///
/// \f$ k \f$-cells follow the canonical `ChainComplex` column order (sorted
/// vertex-id tuples), matching `boundaryMatrix(k)` and `weights(k)`. A vertex
/// carried by no simplex is not a 0-cell and does not appear in \f$ L_0 \f$.
///
/// ## Lorentzian weights
///
/// \f$ W_k \f$ is the **signed** `Simplex::volume()`: a timelike edge
/// (\f$ l^2 < 0 \f$) carries a negative (`SquaredContent`) or imaginary
/// (`Content`) weight, making the inner product indefinite. \f$ L_k \f$ is
/// then generally non-self-adjoint (a discrete d'Alembertian) and is
/// diagonalized with a general `Eigen::ComplexEigenSolver`; eigenvalues may be
/// negative or complex. `RecursiveQuotient::regime()` reports the verified
/// `CertificateRegime`. \f$ \ker L_k \cong H_k \f$ degrades to a pseudo-Hodge
/// decomposition: "harmonic" becomes the small-\f$ |\lambda| \f$ near-kernel,
/// and a representative \f$ h \f$ may be **null**: its norm in the metric
/// that produced it vanishes (\f$ \sum_i W_{k,i}|h_i|^2 \approx 0 \f$ here,
/// \f$ h^\dagger M_k^U h \approx 0 \f$ under the Whitney pencil; see
/// `nullNorms`).
///
/// ## The U(1) connection Laplacian
///
/// `connectionLaplacian` (with `adjacency`, `degree`, `connectionSpectrum` and
/// friends) is a separate operator: the Hermitian U(1)-weighted graph Laplacian
/// \f$ L^{U(1)} = D - A \f$ on the 1-skeleton, from each `Edge`'s complex
/// weight \f$ \text{squaredLength}\cdot e^{i\,\text{phase}} \f$ with the
/// magnitude degree convention \f$ D_{ii} = \sum |\text{squaredLength}| \f$.
/// Its row sums do not vanish, so it is not \f$ L_0 \f$ for any weight. It is
/// indexed over the full sorted vertex-id order (\f$ 0..N-1 \f$), including
/// any lone vertex `ChainComplex` omits, and its eigendecomposition is
/// computed lazily and cached.
///
/// This class is the operator only: it computes no fluxes, cycle bases or Betti
/// numbers (`WilsonLoop` and `ChainComplex` do) and does not gauge-transform
/// the mesh.
class HodgeLaplacian {
  public:
    /// Which quantity the diagonal inner-product weight \f$ W_k \f$ is built
    /// from. Both are complex-valued: this is a choice of inner product, not of
    /// signature. The two give different spectra.
    ///
    /// * `Content` — the \f$ k \f$-content, \f$ W_k = V \f$: the diagonal
    ///   discrete exterior calculus (DEC) star, an edge's weight being
    ///   \f$ \ell = \sqrt{\ell^2} \f$. A timelike edge's weight is imaginary, so
    ///   no null (lightlike) kernel direction exists at any boost.
    /// * `SquaredContent` — the squared \f$ k \f$-content, \f$ W_k = V^2 \f$,
    ///   exactly \f$ \ell^2 \f$ for an edge. Being \f$ \det G/(d!)^2 \f$ it is
    ///   polynomial in the squared edge lengths, hence real and signed: timelike
    ///   cells carry a negative weight, and null kernel directions survive.
    enum class WeightConvention { Content, SquaredContent };

    /// Where the operator's metric comes from.
    ///
    /// Reference for discrete exterior calculus and Whitney forms: Desbrun,
    /// Hirani, Leok and Marsden, "Discrete Exterior Calculus",
    /// arXiv:math/0508341.
    ///
    /// * `DiagonalWeights` — the diagonal per-simplex weights of
    ///   `WeightConvention`. The operator ignores the connection.
    /// * `WhitneyPencil` — the chain-level Whitney Hodge pencil of
    ///   `chainhodge::ChainHodge`, the default: sparse inverse chain metrics
    ///   \f$ M_k \f$ from the complex squared edge lengths, dressed at every
    ///   degree by the \f$ \mathbb{C}^* \f$ links read from the edge phases
    ///   (`chainhodge::Connection::fromSpacetime`). `laplacian(k)` is then the
    ///   dense operator on geometric images
    ///   \f$ L_z = (M_k^U)^{-1} h_k(s,U) M_k^U = (M_k^U)^{-1}\tilde A_k^U \f$,
    ///   whose kernel vectors are the images \f$ z = G_k h \f$, and `pencil(k)`
    ///   returns the pair \f$ (\tilde A_k^U, M_k^U) \f$ it is read from. The
    ///   pencil's reference orientation (ascending vertex id) is checked against
    ///   `ChainComplex::fromSpacetime`'s boundary maps on every assembly and
    ///   refused by name if they differ. `weights(k)` still returns the
    ///   diagonal weights of the convention; the pencil's chain metric is not
    ///   diagonal.
    enum class MetricSource { DiagonalWeights, WhitneyPencil };

    /// Whether spectral-entropy diagnostics retain the complex entries of
    /// \f$ L_k \f$ or perform a phase-blind ablation first.
    /// `IncludeComplexPhase` uses \f$ M=L_k \f$; `IgnoreComplexPhase` uses the
    /// entrywise magnitude \f$ M_{ij}=|(L_k)_{ij}| \f$, changing only the
    /// operator this entropy observable uses, not any stored edge length.
    enum class EntropyPhaseMode { IncludeComplexPhase, IgnoreComplexPhase };

    /// Construct the operator over a triangulation. Edge weights and phases are
    /// read lazily, at the first matrix or spectrum query.
    ///
    /// @param st The triangulation the operator is defined over. Retained for
    ///   the operator's lifetime.
    /// @param weights Which quantity \f$ W_k \f$ is built from. Defaults to
    ///   `SquaredContent`.
    /// @param source Whether the metric is the diagonal per-simplex weights or
    ///   the Whitney Hodge pencil. Defaults to `defaultMetricSource()`.
    explicit HodgeLaplacian(std::shared_ptr<Spacetime> st,
                            WeightConvention weights = defaultWeightConvention(),
                            MetricSource source = defaultMetricSource());

    /// The process-wide default `MetricSource`, read by the constructor's
    /// default argument at the call site. `WhitneyPencil` unless changed.
    [[nodiscard]] static MetricSource defaultMetricSource() noexcept {
      return defaultMetricSource_;
    }
    /// Set the process-wide default `MetricSource`.
    static void setDefaultMetricSource(MetricSource source) noexcept {
      defaultMetricSource_ = source;
    }
    /// The `MetricSource` this instance was built with.
    [[nodiscard]] MetricSource metricSource() const noexcept { return metricSource_; }

    /// Whitney pencil only: the analytic \f$ \partial L_k/\partial\varphi_e \f$
    /// for the multiplicative variation \f$ U_e = e^{i\varphi_e} \f$ of the link
    /// on the edge \f$ (e_a, e_b) \f$, flat row-major \f$ |C_k|^2 \f$.
    /// Identically zero under `DiagonalWeights`.
    [[nodiscard]] std::vector<std::complex<double>> laplacianPhaseGradient(
        int k, std::uint64_t ea, std::uint64_t eb) const;

    /// A Hodge operator paired with the metric of the same source, as flat
    /// row-major \f$ |C_k|\times|C_k| \f$ arrays in the canonical cell order.
    struct MetricPencil {
      /// Under `WhitneyPencil`, \f$ \tilde A_k^U \f$ on geometric images.
      std::vector<std::complex<double>> op{};
      /// Under `WhitneyPencil`, the dressed Whitney mass matrix \f$ M_k^U \f$.
      std::vector<std::complex<double>> metric{};
      /// \f$ |C_k| \f$; 0 above the top dimension.
      int dimension{0};
    };

    /// Whitney pencil only: the pencil \f$ (\tilde A_k^U, M_k^U) \f$ that
    /// `laplacian(k)` is read from, \f$ L_z = (M_k^U)^{-1}\tilde A_k^U \f$, in
    /// the same cell order and stored orientation. Both matrices come from one
    /// assembly of the dressed pencil.
    /// @throws std::logic_error under `DiagonalWeights`, whose metric is the
    ///   diagonal `weights(k)`.
    /// @throws std::runtime_error for \f$ k < 0 \f$.
    [[nodiscard]] MetricPencil pencil(int k) const;

    /// The Kontsevich–Segal allowability margin
    /// \f$ \min_T (\pi - \sum_i |\arg\lambda_i(g_T)|) \f$ of the spacetime's
    /// current squared lengths: positive is allowable, zero is the boundary
    /// (real Lorentzian), negative is not allowable. Independent of the metric
    /// source.
    [[nodiscard]] static double kontsevichSegalMargin(const Spacetime &st);

    /// The process-wide default `WeightConvention`, read by the constructor's
    /// default argument at the call site. `SquaredContent` unless changed. Not
    /// a per-call knob: flipping it mid-run mixes conventions across cached
    /// spectra.
    [[nodiscard]] static WeightConvention defaultWeightConvention() noexcept {
      return defaultWeightConvention_;
    }
    static void setDefaultWeightConvention(WeightConvention convention) noexcept {
      defaultWeightConvention_ = convention;
    }

    /// Weighted adjacency \f$ A \f$ of the **\f$\mathbb{C}^{*}\f$ connection**
    /// operator as a flat row-major \f$ N\times N \f$ complex array over the
    /// full sorted vertex-id order. The stored orientation carries the link
    /// \f$ U = e^{i\varphi} \f$ and the reverse its inverse \f$ U^{-1} \f$, so
    /// \f$ A \f$ is Hermitian exactly when the phases and the squared lengths
    /// are real. Not part of \f$ L_0 \f$; see `connectionLaplacian`.
    [[nodiscard]] std::vector<std::complex<double>> adjacency() const;

    /// Degree vector \f$ (D_{00},\dots,D_{N-1,N-1}) \f$ of the
    /// **\f$\mathbb{C}^{*}\f$ connection** operator, real, length \f$ N \f$
    /// (magnitude convention \f$ D_{ii} = \sum |\text{squaredLength}| \f$, hence
    /// phase-independent). Not part of \f$ L_0 \f$.
    [[nodiscard]] std::vector<double> degree() const;

    /// The **\f$\mathbb{C}^{*}\f$ connection graph Laplacian**
    /// \f$ L^{\mathbb{C}^{*}} = D - A \f$ as a flat row-major \f$ N\times N \f$
    /// array over the full sorted vertex-id order (\f$ N \f$ = every vertex,
    /// including any carried by no simplex). Its off-diagonal uses the signed
    /// complex weight and its diagonal the magnitude, so its row sums do not
    /// vanish and it is not the degree-zero Hodge Laplacian for any \f$ W \f$;
    /// use `laplacian(0)` for that. A nonzero U(1) flux lifts its zero mode. Its
    /// spectrum is invariant under the gauge similarity
    /// \f$ \operatorname{diag}(g)^{-1}(\cdot)\operatorname{diag}(g) \f$ for
    /// every \f$ g:K_0\to\mathbb{C}^{*} \f$. It is Hermitian, positive
    /// semidefinite by Gershgorin, and unitary under
    /// \f$ e^{-iL^{\mathbb{C}^{*}}t} \f$ only for real phases on real weights.
    [[nodiscard]] std::vector<std::complex<double>> connectionLaplacian() const;

    /// Laplacian \f$ L_k \f$ as a flat row-major \f$ |C_k|\times|C_k| \f$ matrix
    /// of complex entries in the canonical `ChainComplex` column order: the
    /// Whitney operator on geometric images under `WhitneyPencil`, or the
    /// operator of the boundary maps and the signed diagonal weights under
    /// `DiagonalWeights`. Generally non-symmetric. `metric = false` selects
    /// unit weights, the combinatorial Laplacian, under either source.
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty for \f$ k \f$ above
    ///   the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> laplacian(int k = 0,
                                                             bool metric = true) const;

    /// Diagonal inner-product weights \f$ W_k \f$ (length \f$ |C_k| \f$) in the
    /// canonical `ChainComplex` column order: the **signed** per-\f$ k \f$-simplex
    /// content `Simplex::volume()` under the active `WeightConvention` (timelike
    /// cells negative or imaginary; degenerate cells fall back to \f$ +1 \f$ so
    /// \f$ W_k \f$ stays invertible). \f$ W_0 = I \f$. Empty for \f$ k < 0 \f$
    /// or \f$ k \f$ above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> weights(int k) const;

    /// Exact analytic gradient of the operator in one edge's squared length,
    /// as a flat row-major \f$ |C_k|\times|C_k| \f$ matrix in the canonical
    /// column order. Under `WhitneyPencil` it is
    /// \f$ \partial L_z/\partial s_e = (M^U)^{-1}[-\partial M^U L_z
    /// + \partial h\,M^U + h\,\partial M^U] \f$. Under `DiagonalWeights` it is
    /// \f$ \partial L_k^{\text{sym}} / \partial \ell^2_e \f$ of the symmetric
    /// metric Hodge Laplacian: with
    /// \f$ L_k = B_k^\top B_k + B_{k+1}B_{k+1}^\top \f$ and
    /// \f$ B_k=\mathrm{diag}(\sqrt{W_{k-1}})\,\partial_k\,\mathrm{diag}(1/\sqrt{W_k}) \f$,
    /// only \f$ W_j=|\!\operatorname{vol}| \f$ depends on \f$ \ell^2 \f$, and
    /// \f$ \partial W_j \f$ is `Simplex::volumeGradient`. At degree zero the
    /// only surviving term is
    /// \f$ -\partial_1 W_1^{-1}(\partial W_1)W_1^{-1}\partial_1^{\dagger} \f$.
    /// All-zero for an edge the complex does not carry; empty for
    /// \f$ k < 0 \f$ or \f$ k \f$ above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> laplacianGradient(
        int k, std::uint64_t edgeA, std::uint64_t edgeB) const;

    /// Von Neumann entropy of the normalized positive Hodge operator
    /// \f[ A_k=M_k^\dagger M_k,\qquad
    ///     \rho_k=A_k/\operatorname{Tr}A_k,\qquad
    ///     S_k=-\operatorname{Tr}(\rho_k\log\rho_k). \f]
    ///
    /// Reference: De Domenico and Biamonte, "Spectral entropies as
    /// information-theoretic tools for complex network comparison",
    /// arXiv:1609.01214.
    ///
    /// `phaseMode` selects \f$ M_k=L_k \f$ or the phase-blind entrywise
    /// \f$ |L_k| \f$ ablation. Empty and identically-zero operators have entropy
    /// zero. Exact zero modes are omitted, so the derivative is taken on the
    /// fixed-rank stratum.
    ///
    /// \f$ L_k \f$ is the operator of this instance's metric source. Under the
    /// default `WhitneyPencil` it is the covariant \f$ h_k(s,U) \f$, so the
    /// entropy sees the connection: \f$ S_k \f$ is a functional of the singular
    /// values of \f$ L_k \f$, which the unitary (U(1)) part of the gauge
    /// similarity preserves, so it depends on a U(1) connection through its
    /// holonomy alone, and its stationarity term is what makes the connection
    /// dynamical. Under `DiagonalWeights` \f$ L_k \f$ is built from the squared
    /// lengths alone and the entropy is independent of \f$ \varphi \f$.
    [[nodiscard]] double spectralEntropy(
        int k, EntropyPhaseMode phaseMode =
                   EntropyPhaseMode::IncludeComplexPhase) const;

    /// Gradient of `spectralEntropy` in the complex squared edge coordinates
    /// \f$ z_e=\ell_e^2 \f$, in `EdgeList` order, with the convention
    /// \f$ h_e=\partial S/\partial\operatorname{Re}z_e
    ///       -i\,\partial S/\partial\operatorname{Im}z_e \f$, so
    /// \f$ \overline h \f$ is the steepest-ascent displacement in the complex
    /// \f$ z \f$ plane. Available at every degree \f$ k\ge0 \f$, and taken of
    /// the operator of this instance's metric source (the analytic
    /// \f$ \partial L_z/\partial z_e \f$ of `laplacianGradient` under
    /// `WhitneyPencil`).
    [[nodiscard]] std::vector<std::complex<double>> spectralEntropyGradient(
        int k, EntropyPhaseMode phaseMode =
                   EntropyPhaseMode::IncludeComplexPhase) const;

    /// \f$ \sum_e |\partial S/\partial z_e|^2 \f$, the entropy-stationarity
    /// residual used by the joint Regge-Hodge objective.
    [[nodiscard]] double spectralEntropyGradientNorm(
        int k, EntropyPhaseMode phaseMode =
                   EntropyPhaseMode::IncludeComplexPhase) const;

    /// Exact analytic Hessian-vector product of the spectral entropy: the
    /// directional derivative
    /// \f$ \dot h_e=\frac{d}{dt}\Big|_{0}h_e(z+t\,v) \f$ of
    /// `spectralEntropyGradient` along \f$ v \f$, in the same `EdgeList` order,
    /// for a real parameter \f$ t \f$, so both \f$ z \f$ and \f$ \bar z \f$
    /// move. Returns an all-zero vector for an empty or identically-zero
    /// operator.
    /// Taken of the operator of this instance's metric source: under
    /// `WhitneyPencil` the second derivative of \f$ L_z \f$ in the squared
    /// lengths comes from `chainhodge::CovariantChainHodge::lengthDirection`
    /// and `covariantOperatorSecondDerivative`.
    /// @throws std::runtime_error if `direction.size()` is not the edge count.
    [[nodiscard]] std::vector<std::complex<double>>
    spectralEntropyGradientDirectionalDerivative(
        int k, const std::vector<std::complex<double>> &direction,
        EntropyPhaseMode phaseMode =
            EntropyPhaseMode::IncludeComplexPhase) const;

    /// The local spectral moments of the Hodge operator,
    /// \f[ \mu_j(x)=\bigl(L_k^{\,j}\bigr)_{xx},\qquad j=1,\dots,m, \f]
    /// one per \f$ k \f$-cell \f$ x \f$ in the canonical column order and per
    /// order, as a flat row-major \f$ |C_k|\times m \f$ array. They are the
    /// local parts of the power sums \f$ p_j=\operatorname{tr}L_k^{\,j}
    /// =\sum_x\mu_j(x) \f$: holomorphic in the complex squared lengths, defined
    /// for a non-normal operator, and needing neither eigenvalues nor a real
    /// projection. \f$ L_k \f$ is the one `spectralEntropy` uses.
    /// @throws std::invalid_argument for \f$ m<1 \f$.
    [[nodiscard]] std::vector<std::complex<double>> localSpectralMoments(
        int k, int orders) const;

    /// The spectral-moment stiffness of the geometric action about a carrier,
    /// \f[ S_M(z)=\tfrac12\sum_{j=1}^{m}\beta_j\sum_x
    ///     \bigl(\mu_j(x;z)-\mu_j(x;z_0)\bigr)^2, \f]
    /// with \f$ \beta_j \f$ = `coefficients[j-1]` and \f$ \mu_j(x;z_0) \f$ =
    /// `reference`, the `localSpectralMoments(k, m)` of the carrier. It is a sum
    /// of local terms, so it is extensive in the size of the complex; it
    /// vanishes with its gradient at the carrier, which therefore stays
    /// stationary; and its Hessian there is
    /// \f$ \sum_j\beta_j\sum_x\nabla\mu_j(x)\,\nabla\mu_j(x)^{T} \f$,
    /// positive semidefinite for non-negative \f$ \beta_j \f$. Holomorphic in
    /// \f$ z \f$ (no conjugation), like the whitepaper's power sums.
    /// @throws std::invalid_argument when `reference` is not
    ///   \f$ |C_k|\times m \f$.
    [[nodiscard]] std::complex<double> spectralMomentStiffness(
        int k, const std::vector<std::complex<double>> &reference,
        const std::vector<double> &coefficients) const;

    /// \f$ \partial S_M/\partial z_e \f$ in `EdgeList` order, the holomorphic
    /// derivative. For a real-valued use of \f$ \operatorname{Re} S_M \f$ it is
    /// also the \f$ h_e=\partial/\partial\operatorname{Re}z_e
    /// -i\,\partial/\partial\operatorname{Im}z_e \f$ of `spectralEntropyGradient`.
    [[nodiscard]] std::vector<std::complex<double>>
    spectralMomentStiffnessGradient(
        int k, const std::vector<std::complex<double>> &reference,
        const std::vector<double> &coefficients) const;

    /// The exact Hessian-vector product
    /// \f$ \sum_f \partial^2 S_M/\partial z_e\partial z_f\,v_f \f$ in
    /// `EdgeList` order: the product rule on the moments, with the exact second
    /// derivative of \f$ L_k \f$ contracted against \f$ v \f$.
    /// @throws std::runtime_error if `direction.size()` is not the edge count.
    [[nodiscard]] std::vector<std::complex<double>>
    spectralMomentStiffnessHessianProduct(
        int k, const std::vector<std::complex<double>> &reference,
        const std::vector<double> &coefficients,
        const std::vector<std::complex<double>> &direction) const;

    /// Entropy of the normalized squared eigenvalue moduli of the
    /// **\f$\mathbb{C}^{*}\f$ connection** Laplacian:
    /// \f[ p_i=\frac{|\lambda_i|^{2}}{\sum_j|\lambda_j|^{2}},\qquad
    ///     S=-\sum_i p_i\log p_i. \f]
    ///
    /// This is the entropy of the 1-skeleton operator; `spectralEntropy` of
    /// the default \f$ h_k(s,U) \f$ sees the connection as well, at every
    /// degree. The weights are eigenvalue
    /// moduli because eigenvalues, unlike singular values, survive the gauge
    /// similarity
    /// \f$ \operatorname{diag}(g)^{-1}(\cdot)\operatorname{diag}(g) \f$,
    /// \f$ g:K_0\to\mathbb{C}^{*} \f$; squared so that for Hermitian \f$ L \f$,
    /// where \f$ |\lambda_i|^2=\sigma_i^2 \f$, \f$ S \f$ agrees with
    /// `spectralEntropy`.
    [[nodiscard]] double connectionSpectralEntropy() const;

    /// Gradient of `connectionSpectralEntropy` in the complex connection phases
    /// \f$ \varphi_e \f$, in `EdgeList` order, in the convention
    /// `spectralEntropyGradient` uses for \f$ z \f$:
    /// \f$ h_e=\partial S/\partial\operatorname{Re}\varphi_e
    ///        -i\,\partial S/\partial\operatorname{Im}\varphi_e \f$, so
    /// \f$ \overline{h} \f$ is the steepest-ascent displacement in the complex
    /// \f$ \varphi \f$ plane and the directional derivative along \f$ v \f$ is
    /// \f$ \sum_e\operatorname{Re}(h_e v_e) \f$. Exact and closed-form. Both
    /// the compact \f$ U(1) \f$ and the non-compact \f$ \mathbb{R}^{+} \f$
    /// components are differentiated.
    [[nodiscard]] std::vector<std::complex<double>>
    connectionSpectralEntropyPhaseGradient() const;

    /// \f$ \sum_e|\partial S/\partial\varphi_e|^2 \f$ — the connection-entropy
    /// stationarity residual, the \f$ \varphi \f$ analogue of
    /// `spectralEntropyGradientNorm`.
    [[nodiscard]] double connectionSpectralEntropyPhaseGradientNorm() const;

    /// Whether \f$ \| L^{U(1)} - (L^{U(1)})^\dagger \| \le \text{tol} \f$
    /// (Frobenius norm) for the **U(1) connection** Laplacian. True by
    /// construction. It says nothing about \f$ L_0 \f$.
    [[nodiscard]] bool isHermitian(double tol = 1e-12) const;

    /// Unitarity residual of the **U(1) connection** time-evolution operator
    /// \f$ U = e^{-iL^{U(1)}t} = V\,\mathrm{diag}(e^{-i\lambda t})\,V^\dagger \f$
    /// formed from its eigendecomposition: \f$ \| U U^\dagger - I \| \f$
    /// (Frobenius), ~0 since that operator is Hermitian.
    [[nodiscard]] double unitarityResidual(double t = 1.0) const;

    /// The **U(1) connection** Laplacian's eigendecomposition as a `Spectrum`
    /// (real ascending eigenvalues, eigenvectors as degree-0 `Cochain`s,
    /// `Spectrum::isHermitian()` true), indexed over the full sorted-id vertex
    /// order.
    [[nodiscard]] Spectrum connectionSpectrum() const;

    /// Eigenvalues of the **U(1) connection** Laplacian (real, ascending),
    /// complex-typed for parity, consistent with `connectionSpectrum()`.
    [[nodiscard]] std::vector<std::complex<double>> connectionEigenvalues() const;

    /// Eigenvectors of the **U(1) connection** Laplacian as a flat row-major
    /// \f$ N\times N \f$ array; column \f$ j \f$ (entries at indices
    /// \f$ iN + j \f$) is the eigenvector for the \f$ j \f$-th ascending
    /// eigenvalue.
    [[nodiscard]] std::vector<std::complex<double>> connectionEigenvectors() const;

    /// Harmonic representatives of the **U(1) connection** Laplacian: the
    /// eigenvectors with \f$ |\lambda| < \text{tol} \f$, as degree-0 `Cochain`s
    /// over the sorted-id vertex order. The count is not \f$ b_0 \f$: a nonzero
    /// U(1) flux lifts it.
    [[nodiscard]] std::vector<Cochain> connectionHarmonics(double tol = 1e-9) const;

    /// The **U(1) connection** harmonic amplitude matrix: the
    /// `connectionHarmonics(tol)` representatives stacked as the **rows** of a
    /// flat row-major \f$ \dim\ker L^{U(1)} \times N \f$ complex array, columns
    /// in the sorted-id vertex order. Empty when that kernel is empty.
    [[nodiscard]] std::vector<std::complex<double>> connectionHarmonicMatrix(
        double tol = 1e-9) const;

    /// The eigendecomposition of \f$ L_k \f$ as a `Spectrum`: eigenvalues
    /// complex, sorted by (Re, Im), `Spectrum::isHermitian()` false,
    /// eigenvectors indexed over the canonical `ChainComplex`
    /// \f$ k \f$-simplex column order. `metric` selects signed-content vs. unit
    /// weights.
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] Spectrum spectrum(int k = 0, bool metric = true) const;

    /// Eigenvalues of \f$ L_k \f$ (complex, sorted by (Re, Im)), a flat view
    /// consistent with `spectrum(k, metric)`. `metric` selects signed-content
    /// vs. unit weights.
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> eigenvalues(int k = 0, bool metric = true) const;

    /// Eigenvectors of \f$ L_k \f$ as a flat row-major
    /// \f$ |C_k|\times|C_k| \f$ array; column \f$ j \f$ (entries at indices
    /// \f$ i|C_k| + j \f$) is the eigenvector for the \f$ j \f$-th eigenvalue,
    /// consistent with the `Cochain`s of `spectrum(k, metric)`. `metric`
    /// selects signed-content vs. unit weights.
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> eigenvectors(int k = 0,
                                                               bool metric = true) const;

    /// Harmonic representatives: the eigenvectors with
    /// \f$ |\lambda| < \text{tol} \f$, a basis for
    /// \f$ \ker L_k \cong H_k \f$, so the count is \f$ b_k \f$ at positive
    /// weights and \f$ b_0 \f$ at \f$ k = 0 \f$ under any weights. Returned as
    /// `Cochain`s over the canonical \f$ k \f$-simplex ordering. `metric`
    /// selects signed-content vs. unit weights.
    /// @throws std::runtime_error for \f$ k < 0 \f$. Empty above the top dimension.
    [[nodiscard]] std::vector<Cochain> harmonics(int k = 0, double tol = 1e-9,
                                                 bool metric = true) const;

    /// The harmonic amplitude matrix: the representatives of
    /// `harmonics(k, tol, metric)`, in spectral order, stacked as the **rows**
    /// of a flat row-major \f$ \dim\ker L_k \times |C_k| \f$ complex array,
    /// columns in the canonical `ChainComplex` \f$ k \f$-cell order. Entry
    /// \f$ [\,r\,|C_k| + c\,] \f$ equals
    /// `harmonics(k, tol, metric)[r].amplitude(c)`. Empty when the kernel is
    /// empty or \f$ k \f$ is above the top dimension. `metric` selects
    /// signed-content vs. unit weights.
    /// @throws std::runtime_error for \f$ k < 0 \f$.
    [[nodiscard]] std::vector<std::complex<double>> harmonicMatrix(
        int k = 0, double tol = 1e-9, bool metric = true) const;

    /// The indefinite norms of the near-kernel representatives in the metric
    /// that produced them, one per column of `harmonics(k, tol, metric)` and
    /// in the same order. Under `WhitneyPencil` the representatives are the
    /// eigenvectors of the pencil \f$ (\tilde A_k^U, M_k^U) \f$ and the norm
    /// is the quadratic form of the dressed Whitney mass on them,
    /// \f$ h^\dagger M_k^U h \f$; under `DiagonalWeights` (and for
    /// `metric = false`, whose weights are the identity) it is
    /// \f$ \langle h,h\rangle_W = \sum_i W_{k,i}|h_i|^2 \f$ with the signed
    /// diagonal \f$ W_k \f$. Either way the sign says which causal character
    /// dominates the direction and a value \f$ \approx 0 \f$ flags a **null**
    /// (lightlike) harmonic; all entries are positive on an all-spacelike
    /// complex. Whether the Whitney form should be the bilinear
    /// \f$ h^T M_k^U h \f$ instead is the open sesquilinear-pairing question;
    /// on a real metric with the trivial connection \f$ M_k \f$ is real
    /// symmetric and the two forms differ only by the conjugation of
    /// \f$ h \f$, so they coincide on real representatives.
    /// @throws std::runtime_error for \f$ k < 0 \f$.
    [[nodiscard]] std::vector<std::complex<double>> nullNorms(
        int k, double tol = 1e-9, bool metric = true) const;

  private:
    std::shared_ptr<Spacetime> st_;
    WeightConvention weightConvention_{WeightConvention::SquaredContent};
    static WeightConvention defaultWeightConvention_;
    MetricSource metricSource_{MetricSource::WhitneyPencil};
    static MetricSource defaultMetricSource_;
    // Whitney pencil state (chain complex, squared lengths, connection, dressed
    // operator), built lazily and rebuilt when the geometry stamp moves.
    struct WhitneyState;
    mutable std::shared_ptr<WhitneyState> whitney_{};
    [[nodiscard]] const WhitneyState &whitneyState() const;
    [[nodiscard]] Eigen::MatrixXcd operatorMatrix(int k, bool metric) const;

    // Stable vertex order: ids_[idx] = vertex id, idToIndex_[id] = idx. Built
    // once in the constructor; only the edge weights/phases are read lazily.
    std::vector<std::uint64_t> ids_{};
    std::unordered_map<std::uint64_t, std::size_t> idToIndex_{};
    std::size_t order_{0};  // N = |V|

    // Lazy, cached Hermitian eigendecomposition of the U(1) connection
    // Laplacian D - A (not L_0).
    mutable bool decomposed_{false};
    mutable std::vector<double> evals_{};               // ascending, length N
    mutable std::vector<std::complex<double>> evecs_{};  // flat N*N, columns

    // General (non-symmetric) eigendecomposition of L_k, cached per (k, metric).
    // Eigenvalues/eigenvectors are complex, sorted ascending by (Re, Im); `wk` is
    // the signed weight diagonal for the null-norm <h,h>_W = sum_i wk[i]|h_i|^2.
    struct SpectrumCache {
      int dim{0};
      std::vector<std::complex<double>> evals{};         // sorted, length |C_k|
      std::vector<std::complex<double>> evecs{};         // flat |C_k|*|C_k|, columns
      std::vector<std::complex<double>> wk{};                          // signed W_k, length |C_k|
    };
    /// The spectrum map, shared across every HodgeLaplacian built on the same
    /// spacetime at the same geometry. The constructor adopts the spacetime's
    /// revision-stamped spectral slot when current, else creates and stores a
    /// fresh map. Entries are keyed by (k, metric, weight convention).
    struct SharedSpectrumMap {
      std::unordered_map<long long, SpectrumCache> map{};
    };
    std::shared_ptr<SharedSpectrumMap> sharedSpectra_{};

    // Throw for k < 0 (no negative-degree chains).
    static void requireNonNegativeDegree(int k);

    // The sorted vertex-id tuples a degree-k Cochain is indexed over.
    // `useVertexSet` returns the full sorted-id vertex order (the U(1)
    // connection basis, length N); otherwise the canonical ChainComplex
    // k-simplex column order (the L_k basis, length |C_k|).
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> cochainOrdering(
        int k, bool useVertexSet) const;

    // Assemble a Spectrum from flat eigenvalues/eigenvectors. `evecsFlat` is
    // row-major dim*dim with entry [i*dim + j] = component i of eigenvector j;
    // `ordering` indexes the components; `hermitian` flags real-ascending.
    static Spectrum makeSpectrum(
        int degree, std::vector<std::vector<std::uint64_t>> ordering,
        const std::vector<std::complex<double>> &evals,
        const std::vector<std::complex<double>> &evecsFlat, int dim,
        bool hermitian);

    // Build/fetch the cached general spectrum of L_k. The key folds in `metric`
    // so the signed-content and combinatorial spectra are cached separately.
    const SpectrumCache &ensureSpectrum(int k, bool metric) const;

    // Assemble the U(1) connection adjacency (flat row-major N*N) and degree
    // (length N) from the current edge weights/phases, in the stable vertex
    // order. Eigen-free signature, so the public header needs no Eigen.
    void assemble(std::vector<std::complex<double>> &A, std::vector<double> &D) const;

    // Build and cache the eigendecomposition of the U(1) connection Laplacian
    // if not already done.
    void ensureDecomposition() const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_HODGELAPLACIAN_H
