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
/// ## One definition, at every degree
///
/// With the oriented integer boundary maps \f[ \partial_k \f] (`ChainComplex`),
/// the diagonal metric weight \f[ W_k \f] on \f[ k \f]-chains (`weights`, built
/// from `Simplex::volume`; \f[ W_0 = I \f]), and the weighted adjoint
/// \f[ \partial_k^{*} = W_k^{-1}\partial_k^{\dagger}W_{k-1}, \f]
/// the operator is
/// \f[ L_k = \partial_{k+1}\partial_{k+1}^{*} + \partial_k^{*}\partial_k
///         = \partial_{k+1}W_{k+1}^{-1}\partial_{k+1}^{\dagger}W_k
///           + W_k^{-1}\partial_k^{\dagger}W_{k-1}\partial_k \f]
/// for **every** \f[ k \geq 0 \f] — degree zero included, with no separate
/// convention. This is the whitepaper's definition
/// (`docs/design/recursive_spectral_fibers_whitepaper.tex`, "The microscopic
/// geometric state") verbatim.
///
/// At \f[ k = 0 \f] the second term is absent (there are no
/// \f[ (-1) \f]-chains), so
/// \f[ L_0 = \partial_1 W_1^{-1}\partial_1^{\dagger} \f]
/// — the graph Laplacian whose off-diagonal entry on the 1-cell
/// \f[ e = (u,v) \f] is \f[ -1/W_1(e) \f] and whose diagonal is
/// \f[ \sum_{e \ni u} 1/W_1(e) \f]. Because \f[ \partial_1^{\dagger} \f] has
/// column sums zero, the **row sums of \f[ L_0 \f] vanish identically**: the
/// constant 0-cochain is harmonic at ANY weights — positive, signed, or complex
/// — so \f[ \dim\ker L_0 = b_0 \f] = the number of connected components,
/// independently of the geometry. That is the degree-zero instance of the
/// discrete Hodge theorem \f[ \ker L_k \cong H_k \f], which holds at every
/// degree for any *positive* weights; `metric = false` selects unit weights
/// (the combinatorial \f[ \partial_{k+1}\partial_{k+1}^{\dagger} +
/// \partial_k^{\dagger}\partial_k \f]) as a same-kernel cross-check, at
/// \f[ k = 0 \f] as at \f[ k \geq 1 \f]. A negative \f[ k \f] throws; a
/// \f[ k \f] beyond the top dimension has no \f[ k \f]-cells and yields empty
/// results.
///
/// ## Regime: Lorentzian, so generally not self-adjoint (§5.6)
///
/// \f[ W_k \f] is the **signed** `Simplex::volume()` — the honest,
/// signature-respecting content in which a timelike edge (\f[ l^2 < 0 \f])
/// carries a negative (`SquaredContent`) or imaginary (`Content`) weight. The
/// inner product is therefore indefinite and the symmetric
/// \f[ W_k^{1/2} \f] similarity breaks (the square root of a negative weight is
/// imaginary), so \f[ L_k \f] is assembled **directly** and is generally
/// **non-self-adjoint** (a discrete d'Alembertian) at every degree, degree zero
/// included: it is diagonalized with a general `Eigen::ComplexEigenSolver`, so
/// eigenvalues may be negative or complex. In the positive metric regime the
/// operator IS self-adjoint in the \f[ W_k \f] inner product; that is a
/// property of the geometry, never a convention imposed on the assembly, and it
/// is **measured**, not asserted — `RecursiveQuotient::regime()` reports the
/// verified `CertificateRegime` for the operator it was handed. On a Lorentzian
/// complex with signed weights \f[ L_0 \f] is routinely indefinite: the
/// 3-cycle with one timelike edge (\f[ l^2 = -\alpha^2 \f]) has
/// \f[ \mathrm{spec}(L_0) = \{0,\,3,\,1 - 2/\alpha^{2}\} \f], negative below
/// \f[ \alpha = \sqrt{2} \f].
///
/// \f[ \ker L_k \cong H_k \f] likewise degrades away from positive weights to a
/// pseudo-Hodge decomposition: "harmonic" becomes the small-\f[ |\lambda| \f]
/// near-kernel, and a near-kernel representative \f[ h \f] may be **null** in
/// the indefinite metric (\f[ \langle h,h\rangle_W = \sum_i W_{k,i}|h_i|^2
/// \approx 0 \f], see `nullNorms`). The one degree-zero statement that survives
/// every weight is the constant: \f[ L_0 \mathbf{1} = 0 \f] exactly.
///
/// \f[ k \f]-cells follow the canonical `ChainComplex` column order (sorted
/// vertex-id tuples) at every degree, so the returned flat row-major matrices
/// are reproducible and align with `boundaryMatrix(k)` and `weights(k)`. A
/// vertex carried by no simplex is not a 0-cell and does not appear in
/// \f[ L_0 \f].
///
/// ## The U(1) connection Laplacian is a DIFFERENT operator
///
/// `connectionLaplacian` (with `adjacency`, `degree`, `connectionSpectrum` and
/// friends) is the **Hermitian U(1)-weighted graph Laplacian**
/// \f[ L^{U(1)} = D - A \f] on the 1-skeleton, assembled from each `Edge`'s
/// complex weight \f[ \text{squaredLength}\cdot e^{i\,\text{phase}} \f]:
///
/// - Adjacency \f[ A_{ij} = \sum_{(i,j)} \text{squaredLength}\cdot e^{i\,\text{phase}} \f],
///   summed over edges between \f[ i \f] and \f[ j \f]; the stored source→target
///   orientation carries \f[ +\text{phase} \f], the reverse carries
///   \f[ -\text{phase} \f], so \f[ A = A^\dagger \f] (Hermitian).
/// - Degree \f[ D_{ii} = \sum |\text{squaredLength}| \f] over incident edges —
///   the **magnitude** convention, which keeps \f[ L^{U(1)} \f] Hermitian and
///   \f[ e^{-iL^{U(1)}t} \f] unitary for complex weights, and makes it
///   diagonally dominant with a non-negative diagonal, hence positive
///   semidefinite by Gershgorin.
///
/// It is NOT \f[ L_0 \f] and is not of the derived form for any weight: on a
/// Lorentzian complex a timelike edge has \f[ l^2 < 0 \f], so the magnitude
/// diagonal and the signed off-diagonal disagree and the row sums do not
/// vanish. It carries the Aharonov--Bohm content that \f[ L_0 \f] cannot —
/// a nonzero U(1) flux lifts its zero mode, whereas \f[ \ker L_0 \f] is always
/// \f[ b_0 \f] — and it is indexed over the FULL sorted vertex-id order
/// (\f[ 0..N-1 \f]), including any lone vertex `ChainComplex` omits. Its
/// consumers are named: `observables::SpectralGap`,
/// `observables::HarmonicDimension`, `KuennethProduct::productCertificate`, and
/// the degree-zero `EigenstateSynthesis` register readout.
///
/// This class is the *operator* only: it does not compute fluxes, cycle bases,
/// or Betti numbers (those are `WilsonLoop` / `ChainComplex`'s job), and it does
/// not gauge-transform the mesh (gauge invariance is exercised by rephasing the
/// edges and rebuilding). The connection operator's Hermitian eigendecomposition
/// is lazily computed (Eigen `SelfAdjointEigenSolver<MatrixXcd>`) and cached.
class HodgeLaplacian {
  public:
    /// Which quantity the diagonal inner-product weight \f[ W_k \f] is built from.
    ///
    /// BOTH are fully Lorentzian and complex-valued; this is a choice of inner
    /// product, not of signature, and neither reintroduces a Euclidean path.
    ///
    /// * `Content` — the \f[ k \f]-content itself, \f[ W_k = V \f]. This is the
    ///   textbook diagonal DEC star: the weight of a \f[ k \f]-simplex is its
    ///   \f[ k \f]-volume. For an edge that is \f[ \ell = \sqrt{\ell^2} \f], so a
    ///   TIMELIKE edge's weight is imaginary. Consequence: spacelike and timelike
    ///   contributions to \f[ \langle h,h\rangle_W \f] are 90° apart in the complex
    ///   plane and can never cancel, so no null (lightlike) kernel direction
    ///   exists at any boost.
    ///
    /// * `SquaredContent` — the squared \f[ k \f]-content, \f[ W_k = V^2 \f]. For an
    ///   edge that is exactly \f[ \ell^2 \f]. Being \f[ \det G/(d!)^2 \f] it is a
    ///   POLYNOMIAL in the squared edge lengths, so on real signed \f[ \ell^2 \f] it
    ///   is real and signed — timelike cells carry a negative weight rather than
    ///   an imaginary one, and genuine null kernel directions survive.
    ///
    /// The two give measurably different spectra. On the 3-cycle with one
    /// timelike edge (\f[ \ell^2 = -\alpha^2 \f]) `Content` gives
    /// \f[ \mathrm{spec}(L_1) = \{0, 3, 1 - 2i/\alpha\} \f] and
    /// \f[ \langle h,h\rangle_W = 2/3 + i\alpha/3 \f], with no null crossing;
    /// `SquaredContent` keeps the weights real-signed and restores one.
    enum class WeightConvention { Content, SquaredContent };

    /// Where the operator's metric comes from.
    ///
    /// * `DiagonalWeights` — the diagonal per-simplex weights of `WeightConvention`
    ///   (the historical path; every existing consumer and default).
    /// * `WhitneyPencil` — the chain-level Whitney Hodge pencil of
    ///   `chainhodge::ChainHodge`: the sparse inverse chain metrics \f[ M_k \f]
    ///   assembled from the complex squared edge lengths, dressed at EVERY
    ///   degree by the \f[ \mathbb{C}^* \f] links read from the edge phases
    ///   (`chainhodge::Connection::fromSpacetime`). `laplacian(k)` is then the
    ///   dense operator on GEOMETRIC IMAGES, \f[ L_z = (M_k^U)^{-1} h_k(s,U) M_k^U \f]
    ///   (the same spectrum as the covariant chain operator; its kernel vectors
    ///   are the images \f[ z = G_k h \f] whose entries are the edge integrals the
    ///   register readouts pair with cycles, specification §4.3 and §6), and
    ///   `laplacianGradient` / `laplacianPhaseGradient` are its analytic
    ///   derivatives. The chain-space operator itself is
    ///   `chainhodge::CovariantChainHodge::covariantOperator`.
    ///   The reference orientation of the pencil (ascending vertex id) is
    ///   checked against `ChainComplex::fromSpacetime`'s boundary maps on
    ///   every assembly and refused by name if they differ. `weights(k)` keeps
    ///   returning the diagonal weights of the convention: the pencil's chain
    ///   metric is not diagonal, and consumers of that metric read it from
    ///   `chainhodge` directly.
    enum class MetricSource { DiagonalWeights, WhitneyPencil };

    /// Whether spectral-entropy diagnostics retain the complex entries of
    /// \f[L_k\f] or perform a phase-blind ablation first.
    ///
    /// `IncludeComplexPhase` uses \f[M=L_k\f]. `IgnoreComplexPhase` uses the
    /// entrywise magnitude \f[M_{ij}=|(L_k)_{ij}|\f]. The latter does NOT replace
    /// an edge length or squared edge length by its magnitude; it changes only
    /// the operator used by this entropy observable.
    enum class EntropyPhaseMode { IncludeComplexPhase, IgnoreComplexPhase };

    /// Construct the operator over a triangulation. Edge weights/phases are read
    /// lazily (at the first matrix/spectrum query), so the spacetime must
    /// outlive the operator; the held `shared_ptr` keeps it alive.
    ///
    /// @param weights Which quantity \f[ W_k \f] is built from (see
    ///   `WeightConvention`). Defaults to `SquaredContent`: it is real and signed
    ///   on real signed \f[ \ell^2 \f], polynomial in the squared edge lengths so
    ///   it carries no branch, and it preserves genuine null kernel directions.
    explicit HodgeLaplacian(std::shared_ptr<Spacetime> st,
                            WeightConvention weights = defaultWeightConvention(),
                            MetricSource source = defaultMetricSource());

    /// The process-wide default `MetricSource`, read by the constructor's
    /// default argument at the call site exactly like `defaultWeightConvention`.
    /// Ships as `DiagonalWeights`, so every historical consumer is unchanged;
    /// `MultiCobordism` selects `WhitneyPencil` for its own operators.
    [[nodiscard]] static MetricSource defaultMetricSource() noexcept {
      return defaultMetricSource_;
    }
    static void setDefaultMetricSource(MetricSource source) noexcept {
      defaultMetricSource_ = source;
    }
    [[nodiscard]] MetricSource metricSource() const noexcept { return metricSource_; }

    /// Whitney pencil only: the analytic \f[ \partial L_k/\partial\varphi_e \f] for
    /// the multiplicative variation \f[ U_e = e^{i\varphi_e} \f] of the link on
    /// the edge \f[ (e_a, e_b) \f], flat row-major \f[ |C_k|^2 \f]. Under
    /// `DiagonalWeights` the operator carries no phase above degree zero and
    /// the result is identically zero.
    [[nodiscard]] std::vector<std::complex<double>> laplacianPhaseGradient(
        int k, std::uint64_t ea, std::uint64_t eb) const;

    /// The Kontsevich–Segal allowability margin
    /// \f[ \min_T (\pi - \sum_i |\arg\lambda_i(g_T)|) \f] of the spacetime's
    /// current squared lengths (positive: allowable; zero: on the boundary, the
    /// real Lorentzian case; negative: not allowable). Measures the geometry
    /// only; independent of the metric source.
    [[nodiscard]] static double kontsevichSegalMargin(const Spacetime &st);

    /// The process-wide default `WeightConvention`, read by the constructor's
    /// default argument AT THE CALL SITE — so every internally-constructed
    /// operator (MultiCobordism's r_U terms, the near-kernel residual, the
    /// register readout, the observables) follows it unless a caller passes an
    /// explicit convention. Ships as `SquaredContent`; an experiment (e.g. the
    /// animation's --hodge-weights flag) may flip it ONCE at startup. Not a
    /// per-call knob: flipping it mid-run mixes conventions across cached
    /// spectra.
    [[nodiscard]] static WeightConvention defaultWeightConvention() noexcept {
      return defaultWeightConvention_;
    }
    static void setDefaultWeightConvention(WeightConvention convention) noexcept {
      defaultWeightConvention_ = convention;
    }

    /// Weighted adjacency \f[ A \f] of the **\f[\mathbb{C}^{*}\f] connection** operator
    /// as a flat row-major \f[ N\times N \f] array of complex entries, over the full
    /// sorted vertex-id order. The stored orientation carries the link
    /// \f[ U = e^{i\varphi} \f] and the reverse carries its INVERSE \f[ U^{-1} \f], so
    /// \f[ A \f] is Hermitian exactly when the phases are real AND the squared lengths
    /// are real (the magnetic-graph case); a complex phase or a complex Lorentzian
    /// weight makes it non-Hermitian by design. Not part of \f[ L_0 \f]; see
    /// `connectionLaplacian`.
    [[nodiscard]] std::vector<std::complex<double>> adjacency() const;

    /// Degree vector \f[ (D_{00},\dots,D_{N-1,N-1}) \f] of the
    /// **\f[\mathbb{C}^{*}\f] connection** operator, real, length \f[ N \f] (magnitude
    /// convention \f[ D_{ii} = \sum |\text{squaredLength}| \f], hence phase-independent).
    /// Not part of \f[ L_0 \f]; see `connectionLaplacian`.
    [[nodiscard]] std::vector<double> degree() const;

    /// The **\f[\mathbb{C}^{*}\f] connection graph Laplacian**
    /// \f[ L^{\mathbb{C}^{*}} = D - A \f] as a flat row-major \f[ N\times N \f] array over
    /// the full sorted vertex-id order (\f[ N \f] = every vertex, including any
    /// carried by no simplex). This is NOT the degree-zero Hodge Laplacian: its
    /// off-diagonal uses the signed complex weight while its diagonal uses the
    /// magnitude, so its row sums do not vanish on a Lorentzian complex and it is
    /// not \f[ \partial_1 W_1^{-1}\partial_1^{\dagger} \f] for any \f[ W \f]. It
    /// is the Aharonov--Bohm operator: the connection twists its HOPPING, and its
    /// zero mode is lifted by a nonzero U(1) flux (which \f[ \ker L_0 = b_0 \f] can
    /// never see). Use `laplacian(0)` for the Hodge operator, which is built from the
    /// lengths alone and is blind to \f[\varphi\f] at every degree.
    ///
    /// A gauge transformation acts on it by the similarity
    /// \f[ \operatorname{diag}(g)^{-1}(\cdot)\operatorname{diag}(g) \f], so its spectrum
    /// is gauge-invariant for every \f[ g:K_0\to\mathbb{C}^{*} \f]. It is Hermitian,
    /// positive semidefinite by Gershgorin and unitary under
    /// \f[ e^{-iL^{\mathbb{C}^{*}}t} \f] only in the compact case — real phases on real
    /// weights — since a complex phase twists by a similarity rather than a unitary.
    [[nodiscard]] std::vector<std::complex<double>> connectionLaplacian() const;

    /// Laplacian \f[ L_k \f] as a flat row-major
    /// \f[ |C_k|\times|C_k| \f] matrix of complex entries in the canonical
    /// `ChainComplex` column order, assembled from the boundary maps and the
    /// signed weights at EVERY degree (degree zero included:
    /// \f[ L_0 = \partial_1 W_1^{-1}\partial_1^{\dagger} \f], whose row sums
    /// vanish identically). Generally non-symmetric — it is the signed-weight
    /// d'Alembertian, not a Hermitian graph Laplacian. `metric = false` selects
    /// unit weights (the combinatorial Laplacian) at every degree, degree zero
    /// included.
    /// @throws std::runtime_error for \f[ k < 0 \f]. Empty for \f[ k \f] above the
    ///   top dimension.
    [[nodiscard]] std::vector<std::complex<double>> laplacian(int k = 0,
                                                             bool metric = true) const;

    /// Diagonal inner-product weights \f[ W_k \f] (length \f[ |C_k| \f]) in the
    /// canonical `ChainComplex` column order: the **signed** per-\f[ k \f]-simplex
    /// content `Simplex::volume()` under the active `WeightConvention` (timelike
    /// cells negative or imaginary; degenerate cells fall back to \f[ +1 \f] so
    /// \f[ W_k \f] stays invertible). \f[ W_0 = I \f] (all ones) — the whitepaper
    /// weight on 0-chains, and what makes the \f[ L_0 \f] row sums vanish. Empty
    /// for \f[ k < 0 \f] or \f[ k \f] above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> weights(int k) const;

    /// Exact analytic gradient \f[ \partial L_k^{\text{sym}} / \partial \ell^2_e \f]
    /// of the symmetric metric Hodge Laplacian (\f[ k \ge 1 \f]) with respect to one
    /// edge's squared length, as a flat \f[ |C_k|\times|C_k| \f] row-major matrix in
    /// the canonical column order. With \f[ L_k = B_k^\top B_k + B_{k+1}B_{k+1}^\top \f],
    /// \f[ B_k=\mathrm{diag}(\sqrt{W_{k-1}})\,\partial_k\,\mathrm{diag}(1/\sqrt{W_k}) \f],
    /// only the inner-product weights \f[ W_j=|\!\operatorname{vol}| \f] depend on
    /// \f[ \ell^2 \f], so \f[ \partial B_k=\mathrm{diag}(a_{k-1})B_k+B_k\,\mathrm{diag}(b_k) \f]
    /// with \f[ a_j=\tfrac{\partial W_j}{2W_j} \f] — and \f[ \partial W_j \f] is the
    /// per-simplex `Simplex::volumeGradient` (signed for the `|vol|` weight). The
    /// degree-generic keystone for the arbitrary-\f[ k \f] \f[ r_U \f] gradient.
    /// Defined at degree zero too, where \f[ W_0 = I \f] is constant and the only
    /// surviving term is
    /// \f[ -\partial_1 W_1^{-1}(\partial W_1)W_1^{-1}\partial_1^{\dagger} \f].
    /// Empty for \f[ k < 0 \f] or an absent edge.
    [[nodiscard]] std::vector<std::complex<double>> laplacianGradient(
        int k, std::uint64_t edgeA, std::uint64_t edgeB) const;

    /// Von Neumann entropy of the normalized positive Hodge operator
    /// \f[ A_k=M_k^\dagger M_k,\qquad
    ///     \rho_k=A_k/\operatorname{Tr}A_k,\qquad
    ///     S_k=-\operatorname{Tr}(\rho_k\log\rho_k). \f]
    ///
    /// `phaseMode` selects \f[M_k=L_k\f] or the phase-blind entrywise
    /// \f[|L_k|\f] ablation. Empty and identically-zero operators have entropy
    /// zero. Exact zero modes are omitted from \f[0\log 0\f] and from the
    /// derivative, which is the derivative on the fixed-rank stratum selected
    /// by the current topology.
    [[nodiscard]] double spectralEntropy(
        int k, EntropyPhaseMode phaseMode =
                   EntropyPhaseMode::IncludeComplexPhase) const;

    /// Gradient of `spectralEntropy` with respect to the COMPLEX squared edge
    /// coordinates \f[z_e=\ell_e^2\f], in `EdgeList` order. The returned
    /// convention is
    /// \f[h_e=\partial S/\partial\operatorname{Re}z_e
    ///       -i\,\partial S/\partial\operatorname{Im}z_e\f], so
    /// \f[\overline h\f] is the steepest-ascent displacement in the complex
    /// \f[z\f] plane. Available at every degree \f[k\ge0\f]: \f[L_k\f] is
    /// holomorphic in \f[z\f] at all of them, degree zero included, because the
    /// weights are polynomial in the squared edge lengths and no modulus enters
    /// the assembly.
    [[nodiscard]] std::vector<std::complex<double>> spectralEntropyGradient(
        int k, EntropyPhaseMode phaseMode =
                   EntropyPhaseMode::IncludeComplexPhase) const;

    /// \f[\sum_e |\partial S/\partial z_e|^2\f], the entropy-stationarity
    /// residual used by the joint Regge-Hodge objective.
    [[nodiscard]] double spectralEntropyGradientNorm(
        int k, EntropyPhaseMode phaseMode =
                   EntropyPhaseMode::IncludeComplexPhase) const;

    /// The **exact analytic Hessian-vector product** of the spectral entropy:
    /// the directional derivative
    /// \f[ \dot h_e=\frac{d}{dt}\Big|_{0}h_e(z+t\,v) \f] of
    /// `spectralEntropyGradient` along a direction \f[ v \f] given in the same
    /// `EdgeList` order, for a REAL parameter \f[ t \f] (so both \f[ z \f] and
    /// \f[ \bar z \f] move). This is what the descent direction of
    /// \f[ \|\nabla_z S\|^2 \f] requires, and it is closed-form — the exactness
    /// contract admits no finite-difference direction.
    ///
    /// Three exact ingredients compose it: the simplex volume Hessian
    /// (`Simplex::volumeGradientDirectionalDerivative`, exact because the Gram
    /// matrix is linear in \f[ \ell^2 \f]), the resulting second derivative of
    /// \f[ L_k \f] contracted against \f[ v \f], and the Daleckii--Krein
    /// derivative of \f[ C=\partial S/\partial A \f] on the fixed-rank stratum
    /// the value and first derivative already select. Cost is one extra
    /// eigenbasis contraction plus the same sparse per-edge assembly the
    /// gradient performs, so it replaces two full analytic-gradient
    /// evaluations rather than adding to them.
    ///
    /// Returns an all-zero vector for an empty or identically-zero operator,
    /// matching `spectralEntropyGradient`. A `direction` whose length differs
    /// from the edge count is rejected.
    /// @throws std::runtime_error if `direction.size()` is not the edge count.
    [[nodiscard]] std::vector<std::complex<double>>
    spectralEntropyGradientDirectionalDerivative(
        int k, const std::vector<std::complex<double>> &direction,
        EntropyPhaseMode phaseMode =
            EntropyPhaseMode::IncludeComplexPhase) const;

    /// Entropy of the normalized SQUARED EIGENVALUE MODULI of the
    /// **\f[\mathbb{C}^{*}\f] connection** Laplacian:
    /// \f[ p_i=\frac{|\lambda_i|^{2}}{\sum_j|\lambda_j|^{2}},\qquad
    ///     S=-\sum_i p_i\log p_i. \f]
    ///
    /// This is the entropy of the operator the connection ACTS on. \f[ L_k \f]
    /// is blind to \f[\varphi\f] at every degree, so no Hodge entropy can see
    /// the connection at all; this one can, because a nonzero flux lifts the
    /// zero mode \f[ \ker L_0=b_0 \f] can never register.
    ///
    /// ## Why the weights are eigenvalue moduli and not M^dagger M
    ///
    /// Do not "simplify" this to
    /// \f[ A=M^\dagger M,\ \rho=A/\operatorname{Tr}A \f] on the grounds that the
    /// two agree in the Hermitian limit. That form is a functional of the
    /// SINGULAR values of \f[ M \f], and singular values are preserved by
    /// UNITARY similarity only.
    ///
    /// A gauge transformation acts here by
    /// \f[ \operatorname{diag}(g)^{-1}(\cdot)\operatorname{diag}(g) \f] for
    /// \f[ g:K_0\to\mathbb{C}^{*} \f]. Since
    /// \f[ \mathbb{C}^{*}\cong U(1)\times\mathbb{R}^{+} \f], that similarity is
    /// unitary exactly when \f[ g \f] is a pure phase. Under complex
    /// \f[ \varphi \f] this operator is explicitly NON-NORMAL, which is where
    /// eigenvalues and singular values part company: the EIGENvalues are
    /// invariant under the full \f[ \mathbb{C}^{*} \f] similarity, the singular
    /// values are not. Measured on a fluxed \f[ S^4 \f] host, the
    /// \f[ M^\dagger M \f] form drifts \f[ 1.6\times10^{-15} \f] under real
    /// \f[ \chi \f] but \f[ 4.9\times10^{-3} \f] under complex \f[ \chi \f],
    /// while the form used here stays at \f[ 4.4\times10^{-16} \f] under both,
    /// with the gradient identity at \f[ 3.3\times10^{-15} \f]. A term that is not
    /// \f[ \mathbb{C}^{*} \f]-invariant is the wrong functional for a
    /// \f[ \mathbb{C}^{*} \f] connection, so gauge invariance is STRUCTURAL
    /// here rather than something a test has to keep re-confirming.
    ///
    /// ## Why the weights are SQUARED
    ///
    /// The square is what makes this degrade gracefully. Eigenvalues are
    /// invariant under the similarity at any power, so \f[|\lambda|\f] and
    /// \f[|\lambda|^2\f] are equally \f[\mathbb{C}^{*}\f]-invariant — but only
    /// the square reduces to the Hodge term's own functional. For Hermitian
    /// \f[ L \f] the eigenvalues of \f[ A=L^\dagger L \f] are exactly
    /// \f[ |\lambda_i|^2=\sigma_i^2 \f], so this IS the von Neumann entropy of
    /// \f[ A/\operatorname{Tr}A \f] there — measured agreement
    /// \f[ 4.4\times10^{-16} \f] — and the two definitions separate only where
    /// the operator stops being normal, by \f[ 1.8\times10^{-3} \f] once a
    /// complex phase makes it non-normal. The
    /// unsquared weight \f[ p_i\propto|\lambda_i| \f] is gauge-invariant too but
    /// lands on the entropy of \f[ |L|/\operatorname{Tr}|L| \f] instead, which
    /// differs from the Hodge form by \f[ 8.7\times10^{-2} \f] on a Hermitian
    /// connection operator. Both properties at once is why the weight is
    /// \f[|\lambda|^2\f].
    [[nodiscard]] double connectionSpectralEntropy() const;

    /// Gradient of `connectionSpectralEntropy` with respect to the COMPLEX
    /// connection phases \f[ \varphi_e \f], in `EdgeList` order, in the same
    /// convention `spectralEntropyGradient` uses for \f[ z \f]:
    /// \f[ h_e=\partial S/\partial\operatorname{Re}\varphi_e
    ///        -i\,\partial S/\partial\operatorname{Im}\varphi_e \f], so
    /// \f[ \overline{h} \f] is the steepest-ascent displacement in the complex
    /// \f[ \varphi \f] plane and the directional derivative along a displacement
    /// \f[ v \f] is \f[ \sum_e\operatorname{Re}(h_e v_e) \f].
    ///
    /// \f[ L^{\mathbb{C}^{*}} \f] is holomorphic in \f[ \varphi \f] — the stored
    /// orientation carries \f[ e^{i\varphi} \f] and the reverse its INVERSE
    /// \f[ e^{-i\varphi} \f], never its conjugate — so each simple eigenvalue is
    /// holomorphic too, with the standard non-Hermitian perturbation formula
    /// \f[ d\lambda_k=u_k^\dagger(dL)v_k/(u_k^\dagger v_k) \f] for left/right
    /// eigenvectors \f[ u_k,v_k \f]. \f[ S \f] itself is NOT holomorphic —
    /// \f[ |\lambda|^2=\lambda\overline{\lambda} \f] is not — but the
    /// non-holomorphy enters only through that squared modulus, in closed form,
    /// so this is still exact rather than a real-parameter approximation. With
    /// \f[ \beta_k=2\,\frac{\partial S}{\partial|\lambda_k|^{2}}
    ///            \overline{\lambda_k} \f] and
    /// \f[ P=V\operatorname{diag}(\beta)V^{-1} \f], each edge touches exactly two
    /// entries of the operator, so the assembly is \f[ O(1) \f] per edge given
    /// that one shared \f[ P \f]. No division by \f[ |\lambda| \f] appears,
    /// which is a second reason to prefer the squared weight.
    ///
    /// BOTH components are differentiated. Only the compact \f[ U(1) \f] part
    /// has winding and quantizes, but the non-compact \f[ \mathbb{R}^{+} \f] part
    /// rescales the hopping and is a real degree of freedom; excluding it would
    /// IMPOSE an irrelevance that must instead be measured.
    [[nodiscard]] std::vector<std::complex<double>>
    connectionSpectralEntropyPhaseGradient() const;

    /// \f[ \sum_e|\partial S/\partial\varphi_e|^2 \f] — the connection-entropy
    /// stationarity residual, the \f[ \varphi \f] analogue of
    /// `spectralEntropyGradientNorm`.
    [[nodiscard]] double connectionSpectralEntropyPhaseGradientNorm() const;

    /// Whether \f[ \| L^{U(1)} - (L^{U(1)})^\dagger \| \le \text{tol} \f]
    /// (Frobenius norm) for the **U(1) connection** Laplacian. True by
    /// construction. It says nothing about \f[ L_0 \f], which is complex
    /// symmetric (hence non-Hermitian) as soon as a weight is complex.
    [[nodiscard]] bool isHermitian(double tol = 1e-12) const;

    /// Unitarity residual of the **U(1) connection** time-evolution operator
    /// \f[ U = e^{-iL^{U(1)}t} = V\,\mathrm{diag}(e^{-i\lambda t})\,V^\dagger \f]
    /// formed from its eigendecomposition: returns \f[ \| U U^\dagger - I \| \f]
    /// (Frobenius). ~0, that operator being Hermitian.
    [[nodiscard]] double unitarityResidual(double t = 1.0) const;

    /// The **U(1) connection** Laplacian's eigendecomposition as a `Spectrum`
    /// (real ascending eigenvalues + eigenvectors as degree-0 `Cochain`s;
    /// `Spectrum::isHermitian()` is true), indexed over the full sorted-id vertex
    /// order. Lazily computed and cached.
    [[nodiscard]] Spectrum connectionSpectrum() const;

    /// Eigenvalues of the **U(1) connection** Laplacian (real, ascending),
    /// complex-typed for parity, consistent with `connectionSpectrum()`.
    [[nodiscard]] std::vector<std::complex<double>> connectionEigenvalues() const;

    /// Eigenvectors of the **U(1) connection** Laplacian as a flat row-major
    /// \f[ N\times N \f] array; column \f[ j \f] (entries at indices
    /// \f[ iN + j \f]) is the eigenvector for the \f[ j \f]-th ascending
    /// eigenvalue.
    [[nodiscard]] std::vector<std::complex<double>> connectionEigenvectors() const;

    /// Harmonic representatives of the **U(1) connection** Laplacian: the
    /// eigenvectors with \f[ |\lambda| < \text{tol} \f], as degree-0 `Cochain`s
    /// over the sorted-id vertex order. Unlike \f[ \ker L_0 \f] this count is
    /// NOT \f[ b_0 \f] — a nonzero U(1) flux lifts it.
    [[nodiscard]] std::vector<Cochain> connectionHarmonics(double tol = 1e-9) const;

    /// The **U(1) connection** harmonic amplitude matrix: the
    /// `connectionHarmonics(tol)` representatives stacked as the ROWS of a flat
    /// row-major \f[ \dim\ker L^{U(1)} \times N \f] complex array, columns in the
    /// sorted-id vertex order. Empty when that kernel is empty.
    [[nodiscard]] std::vector<std::complex<double>> connectionHarmonicMatrix(
        double tol = 1e-9) const;

    /// The eigendecomposition of \f[ L_k \f] as a `Spectrum`. \f[ L_k \f] is the
    /// signed-weight d'Alembertian at every degree, generally non-self-adjoint,
    /// so the eigenvalues are complex, sorted by (Re, Im), and
    /// `Spectrum::isHermitian()` is false. `metric` selects signed-content vs.
    /// unit weights. The eigenvectors are indexed over the canonical
    /// `ChainComplex` \f[ k \f]-simplex column order at every degree, degree zero
    /// included.
    /// @throws std::runtime_error for \f[ k < 0 \f]. Empty above the top dimension.
    [[nodiscard]] Spectrum spectrum(int k = 0, bool metric = true) const;

    /// Eigenvalues of \f[ L_k \f] (complex, sorted by (Re, Im)), a flat view
    /// consistent with `spectrum(k, metric)`. `metric` selects signed-content vs.
    /// unit weights.
    /// @throws std::runtime_error for \f[ k < 0 \f]. Empty above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> eigenvalues(int k = 0, bool metric = true) const;

    /// Eigenvectors of \f[ L_k \f] as a flat row-major
    /// \f[ |C_k|\times|C_k| \f] array; column \f[ j \f] (entries at indices
    /// \f[ i|C_k| + j \f]) is the eigenvector for the \f[ j \f]-th eigenvalue —
    /// a flat view consistent with the `Cochain`s of `spectrum(k, metric)`.
    /// `metric` selects signed-content vs. unit weights.
    /// @throws std::runtime_error for \f[ k < 0 \f]. Empty above the top dimension.
    [[nodiscard]] std::vector<std::complex<double>> eigenvectors(int k = 0,
                                                               bool metric = true) const;

    /// Harmonic representatives: the eigenvectors with \f[ |\lambda| < \text{tol} \f]
    /// (a basis for \f[ \ker L_k \cong H_k \f], so the count is the harmonic
    /// dimension \f[ = b_k \f] at positive weights; at \f[ k = 0 \f] the constant
    /// is among them at ANY weights, so \f[ \dim\ker L_0 = b_0 \f] always), as
    /// `Cochain`s over the canonical \f[ k \f]-simplex ordering. `metric` selects
    /// signed-content vs. unit weights.
    /// @throws std::runtime_error for \f[ k < 0 \f]. Empty above the top dimension.
    [[nodiscard]] std::vector<Cochain> harmonics(int k = 0, double tol = 1e-9,
                                                 bool metric = true) const;

    /// The harmonic amplitude matrix: the same representatives as
    /// `harmonics(k, tol, metric)` — the eigenvectors with
    /// \f[ |\lambda| < \text{tol} \f], in spectral order — stacked
    /// as the **rows** of a flat row-major \f[ \dim\ker L_k \times |C_k| \f]
    /// complex array, columns in the canonical `ChainComplex`
    /// \f[ k \f]-cell order the `Cochain`s index. One call replaces the
    /// per-cell `amplitudeFor` round-trips a register layer makes to read its
    /// harmonics; entry \f[ [\,r\,|C_k| + c\,] \f] equals
    /// `harmonics(k, tol, metric)[r].amplitude(c)` exactly. Empty when the
    /// kernel is empty (\f[ b_k = 0 \f]) or \f[ k \f] is above the top
    /// dimension. `metric` selects signed-content vs. unit weights.
    /// @throws std::runtime_error for \f[ k < 0 \f].
    [[nodiscard]] std::vector<std::complex<double>> harmonicMatrix(
        int k = 0, double tol = 1e-9, bool metric = true) const;

    /// The indefinite norms \f[ \langle h,h\rangle_W = \sum_i W_{k,i}|h_i|^2 \f]
    /// (signed \f[ W_k \f]) of the near-kernel representatives, one per column of
    /// `harmonics(k, tol, metric)` and in the same order. A value
    /// \f[ \approx 0 \f] flags a **null** harmonic (a lightlike kernel direction);
    /// all entries are positive on an all-spacelike complex.
    /// @throws std::runtime_error for \f[ k < 0 \f].
    [[nodiscard]] std::vector<std::complex<double>> nullNorms(
        int k, double tol = 1e-9, bool metric = true) const;

  private:
    std::shared_ptr<Spacetime> st_;
    WeightConvention weightConvention_{WeightConvention::SquaredContent};
    static WeightConvention defaultWeightConvention_;
    MetricSource metricSource_{MetricSource::DiagonalWeights};
    static MetricSource defaultMetricSource_;
    // The Whitney pencil state (chain complex, squared lengths, connection,
    // dressed operator), built lazily and rebuilt when the geometry stamp moves.
    struct WhitneyState;
    mutable std::shared_ptr<WhitneyState> whitney_{};
    [[nodiscard]] const WhitneyState &whitneyState() const;
    [[nodiscard]] Eigen::MatrixXcd operatorMatrix(int k, bool metric) const;

    // Stable vertex order: ids_[idx] = vertex id, idToIndex_[id] = idx. Built
    // once in the constructor (the vertex set is fixed for the operator's life;
    // only the edge weights/phases are read lazily).
    std::vector<std::uint64_t> ids_{};
    std::unordered_map<std::uint64_t, std::size_t> idToIndex_{};
    std::size_t order_{0};  // N = |V|

    // Lazy, cached Hermitian eigendecomposition of the U(1) CONNECTION
    // Laplacian D - A (not L_0).
    mutable bool decomposed_{false};
    mutable std::vector<double> evals_{};               // ascending, length N
    mutable std::vector<std::complex<double>> evecs_{};  // flat N*N, columns

    // General (non-symmetric) eigendecomposition of the signed-weight d'Alembertian
    // L_k, cached per (k, metric). Eigenvalues/eigenvectors are complex and sorted
    // ascending by (Re, Im); `wk` is the signed weight diagonal kept for the
    // indefinite null-norm <h,h>_W = sum_i wk[i] |h_i|^2.
    struct SpectrumCache {
      int dim{0};
      std::vector<std::complex<double>> evals{};         // sorted, length |C_k|
      std::vector<std::complex<double>> evecs{};         // flat |C_k|*|C_k|, columns
      std::vector<std::complex<double>> wk{};                          // signed W_k, length |C_k|
    };
    /// The spectrum map, SHARED across every HodgeLaplacian built on the same
    /// spacetime at the same geometry (#688): the residual/readout path
    /// constructs a fresh operator per call (~275x per objective evaluation,
    /// measured on #683), and a per-instance map re-ran the dense
    /// ComplexEigenSolver each time. The constructor adopts the spacetime's
    /// revision-stamped spectral slot when current, else creates a fresh map
    /// and stores it; entries are keyed by (k, metric, weight convention), so
    /// instances with different conventions share the map without collisions.
    /// The deliberately-uncached ``apply``/``residual`` honesty paths and the
    /// U(1) connection Hermitian decomposition below are untouched.
    struct SharedSpectrumMap {
      std::unordered_map<long long, SpectrumCache> map{};
    };
    std::shared_ptr<SharedSpectrumMap> sharedSpectra_{};

    // Throw for k < 0 (no negative-degree chains).
    static void requireNonNegativeDegree(int k);

    // The sorted vertex-id tuples a degree-k Cochain is indexed over.
    // `useVertexSet` returns the full sorted-id vertex order (the U(1)
    // connection basis, length N); otherwise the canonical ChainComplex
    // k-simplex column order (the L_k basis at EVERY degree, length |C_k|).
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> cochainOrdering(
        int k, bool useVertexSet) const;

    // Assemble a Spectrum from flat eigenvalues/eigenvectors. `evecsFlat` is
    // row-major dim*dim with entry [i*dim + j] = component i of eigenvector j
    // (column j); `ordering` indexes the components; `hermitian` flags the
    // real-ascending regime.
    static Spectrum makeSpectrum(
        int degree, std::vector<std::vector<std::uint64_t>> ordering,
        const std::vector<std::complex<double>> &evals,
        const std::vector<std::complex<double>> &evecsFlat, int dim,
        bool hermitian);

    // Build/fetch the cached general spectrum of the signed-weight
    // d'Alembertian L_k, at every degree. The key folds in `metric` so the
    // signed-content and combinatorial spectra are cached separately.
    const SpectrumCache &ensureSpectrum(int k, bool metric) const;

    // Assemble the U(1) connection adjacency (flat row-major N*N) and degree
    // (length N) from the current edge weights/phases, using the stable vertex
    // order. Kept Eigen-free in its signature so the public header carries no
    // Eigen dependency.
    void assemble(std::vector<std::complex<double>> &A, std::vector<double> &D) const;

    // Build and cache the eigendecomposition of the U(1) connection Laplacian
    // if not already done.
    void ensureDecomposition() const;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_HODGELAPLACIAN_H
