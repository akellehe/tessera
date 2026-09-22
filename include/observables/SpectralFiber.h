// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SPECTRALFIBER_H
#define TESSERA_OBSERVABLES_SPECTRALFIBER_H

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cobordism/Certificate.h"
#include "cobordism/HodgeLaplacian.h"
#include "observables/PersistentModularity.h"
#include "observables/Record.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::cobordism {
  class AnalyticCache;
}
namespace tessera::observables {

/// Configuration of the spectral-band detector and tracker.
///
/// Reference: Lim, "Hodge Laplacians on graphs", arXiv:1507.05379.
///
/// All thresholds are analysis parameters: they select which bands are
/// certified, never which eigenvalues exist, and none is a Betti-number oracle
/// — the zero band is found by the same relative gap rule as every other band.
struct SpectralFiberConfig {
  /// Form degrees enumerated by
  /// :func:`SpectralFiberTracker::enumerateOnComponents`. The detector
  /// enumerates whatever ranks the gap rule produces at these degrees; it never
  /// requests a particular rank.
  std::vector<int> degrees{0, 1, 2};
  /// Relative band-grouping width: consecutive eigenvalues (sorted by
  /// (Re, Im)) belong to one band when their distance is at most
  /// `groupingTolerance * scale`, `scale` the spectral scale
  /// (max |eigenvalue|, or 1 for an identically zero operator).
  double groupingTolerance = 1e-8;
  /// Isolation floor: a band is certified only when its separation from the
  /// nearest discarded eigenvalue in the complex plane
  /// (`SpectralBandCertificate::nearestDiscardedSeparation`) is at least
  /// `minRelativeGap * scale`. A closing gap returns an uncertified band rather
  /// than a discontinuous identity change.
  double minRelativeGap = 1e-6;
  /// A certified band's separation must also dominate its own spread:
  /// separation >= gapDominance * (in-band eigenvalue spread).
  double gapDominance = 4.0;
  /// Certification cap on the relative residuals (eigen, left, projector
  /// idempotency), all measured relative to the operator's Frobenius norm.
  double residualTolerance = 1e-9;
  /// Certification cap on the weighted Gram / signature defect
  /// epsilon_G = ||Phi^dagger W Phi - J||.
  double gramDefectTolerance = 1e-8;
  /// Certification cap on the band projector norm \f$ \|P\|_2 \f$, Kato's
  /// condition number of the spectral projector: 1 for an orthogonal projector,
  /// larger the more oblique the band. This is the gauge-invariant conditioning
  /// of the band. The frame condition number
  /// (`SpectralBandCertificate::frameConditionNumber`) depends on the in-band
  /// basis choice and is reported rather than capped.
  double projectorNormCap = 1e8;
  /// Certification cap on the band's localization excess
  /// (`SpectralBandCertificate::localizationExcess`), which enforces the
  /// acceptance conjunct "a localized spectral projector with stable rank". The
  /// excess is 0 for a band as concentrated as its rank permits and 1 for a
  /// perfectly delocalized one, so the default 0.5 certifies a band no more
  /// than halfway from maximally localized to fully spread. Setting it to 1.0
  /// accepts any measured localization; an unmeasured NaN still fails.
  double maxLocalizationExcess = 0.5;
  /// Dimension at and above which the self-adjoint paths switch from the dense
  /// solve to the sparse block solve; matches
  /// `DenseReference::kDefaultCrossoverDimension`.
  int denseCrossover = 512;
  /// Number of lowest eigenpairs the sparse block path computes. The read is
  /// marked `truncated` when this covers only part of the spectrum; the first
  /// uncovered Ritz value bounds the last covered band's upper gap.
  int requestedEigenpairs = 32;
  /// Extra Ritz vectors carried by the sparse block solve beyond
  /// `requestedEigenpairs`. Improves convergence and supplies the shield value
  /// bounding the last covered gap.
  int oversample = 8;
  /// Sparse block solve iteration cap and relative residual target.
  int maxSolverIterations = 400;
  double solverTolerance = 1e-12;
  /// Seed of the deterministic sparse-path start block.
  std::uint64_t solverSeed = 0;
  /// Minimum subspace overlap for a certified track continuation in
  /// :func:`SpectralFiberTracker::matchFibers`.
  double trackOverlapThreshold = 0.5;
  /// When true and the component is below `denseCrossover`, every solve is
  /// cross-checked against the independent `DenseReference` kernel and the
  /// measured deviation is recorded on the certificate
  /// (`Certificate::denseReferenceError`).
  bool crossValidateDense = false;
  /// Chain-level Whitney pencil path only
  /// (`HodgeLaplacian::MetricSource::WhitneyPencil`): the trapezoidal node count
  /// of the circular Riesz contour drawn around each band, and the relative
  /// tolerance below which the band's bilinear pairing `B_C` is declared
  /// isotropic, i.e. an exceptional point with no left frame.
  int contourNodes = 64;
  double isotropyTolerance = 1e-10;
};

/// # SpectralBandCertificate
///
/// The certification record of one spectral band: what was measured about the
/// band, in which metric regime, and whether the band is certified. Quantities
/// that were not measured are quiet NaN
/// (`cobordism::Certificate::kUnmeasured`), never zero.
///
/// A degenerate band is a single object of rank at least 2. An unexplained
/// multiplicity is reported as its rank and is never labeled a Kähler-Dirac
/// taste, a flavor, or a color; rank-three selection belongs to a classifier,
/// not to this detector. A negative Krein signature is likewise a certificate,
/// not an antiparticle identification.
struct SpectralBandCertificate {
  /// Form degree k of the restricted Hodge operator the band lives on.
  int degree = 0;
  /// Band rank r = number of eigenvalues in the band (with multiplicity).
  std::size_t rank = 0;
  /// Distance from the band to the sort-adjacent eigenvalue below and above it
  /// (complex modulus, over the (Re, Im)-sorted spectrum). Diagnostics only:
  /// with a genuinely complex spectrum the sorted neighbour need not be the
  /// nearest eigenvalue in the plane, so the isolation conjunct is enforced on
  /// `nearestDiscardedSeparation` instead. Infinite when no eigenvalue exists on
  /// that side; NaN when the side is unknown, as at the truncated sparse top.
  double lowerGap = std::numeric_limits<double>::infinity();
  double upperGap = std::numeric_limits<double>::infinity();
  /// The band gap: the distance in the complex plane from the band to the
  /// nearest eigenvalue outside it,
  /// \f$ \min_{j \notin C, i \in C} |\lambda_i - \lambda_j| \f$, which on a
  /// truncated sparse read is additionally bounded by the shield value.
  /// Infinite when nothing was discarded; NaN when an uncovered side leaves it
  /// unknown. Acceptance gates on this quantity.
  double nearestDiscardedSeparation = std::numeric_limits<double>::infinity();
  /// Inverse participation ratio of the band projector's diagonal density
  /// p_i = |P_ii| / sum_j |P_jj|: localization = sum_i p_i^2, in
  /// [1/n, 1/rank] — 1/rank fully localized on `rank` cells (1 for a
  /// rank-one band on a single cell), 1/n perfectly spread.
  /// Gauge-invariant (reads only the projector) and relabeling-invariant.
  double localization = std::numeric_limits<double>::quiet_NaN();
  /// Effective support fraction \f$ n_{\mathrm{eff}}/n = 1/(n\,\mathrm{loc})
  /// \f$ in [rank/n, 1]: 1 for a perfectly delocalized band (uniform projector
  /// diagonal), rank/n for a band concentrated on `rank` cells. Gauge- and
  /// relabeling-invariant and comparable across dimensions, but not across
  /// ranks, since a rank-r band cannot read below r/n.
  double localizationSupportFraction = std::numeric_limits<double>::quiet_NaN();
  /// The localization datum acceptance gates on: the rank-normalized excess
  /// \f$ (n_{\mathrm{eff}} - r)/(n - r) \f$ in [0, 1]. It is 0 when the band
  /// is as concentrated as a rank-r projector can be and 1 when it is perfectly
  /// delocalized. Defined as 0 when the band spans the whole operator
  /// (\f$ n = r \f$), where there is no room left to be localized in.
  double localizationExcess = std::numeric_limits<double>::quiet_NaN();
  /// Idempotency defect ||P^2 - P||_F / max(1, ||P||_F).
  double projectorResidual = std::numeric_limits<double>::quiet_NaN();
  /// Eigenvalue residual ||L Phi - Phi Lambda||_F / ||L||_F on the
  /// eigen-paired right frame.
  double eigenResidual = std::numeric_limits<double>::quiet_NaN();
  /// Left-frame residual ||L^dagger Y - Y Lambda^dagger||_F / ||L||_F with
  /// Y = W^dagger Psi the Euclidean left frame. Equal to `eigenResidual` on the
  /// self-adjoint path, where the frames coincide.
  double leftResidual = std::numeric_limits<double>::quiet_NaN();
  /// Weighted Gram defect: ||Phi^dagger W Phi - J||_F with J = diag(I_p, -I_q)
  /// in the self-adjoint and Krein-normalizable regimes;
  /// ||Psi^dagger W Phi - I||_F on the biorthogonal path.
  double gramDefect = std::numeric_limits<double>::quiet_NaN();
  /// Band projector norm \f$ \|P\|_2 \f$: Kato's condition number of the
  /// spectral projector, 1 for an orthogonal projector and larger the more
  /// oblique the band. It depends only on the band's ranges, so it is
  /// gauge-invariant, and it is what `SpectralFiberConfig::projectorNormCap`
  /// caps.
  double projectorNorm = std::numeric_limits<double>::quiet_NaN();
  /// Frame condition number, reported in the non-normal regime:
  /// \f$ \max(\kappa(\Phi), \kappa(\Psi)) \f$ with
  /// \f$ \kappa(X) = \sqrt{\lambda_{\max}/\lambda_{\min}} \f$ of
  /// \f$ X^\dagger |W| X \f$, the Riesz condition of each reported frame in
  /// the same \f$ |W| \f$ metric the Gram certificate uses. Exactly 1 on the
  /// self-adjoint path, where a W-orthonormal frame is perfectly conditioned,
  /// and larger the more oblique the matched pair. Distinct from
  /// `projectorNorm`: it is a property of the reported frames, so a different
  /// in-band basis changes it. Acceptance caps the gauge-invariant projector
  /// norm and reports this one.
  double frameConditionNumber = std::numeric_limits<double>::quiet_NaN();
  /// Krein inertia of Phi^dagger W Phi: the positive and negative eigenvalue
  /// counts (p, q). Neutral directions are rank - p - q, nonzero only when the
  /// W-Gram is singular, as for complex-eigenvalue Krein bands. In the positive
  /// regime p = rank and q = 0. A negative signature is a certificate, not an
  /// antiparticle identification.
  int positiveSignature = 0;
  int negativeSignature = 0;
  /// Chain-level Whitney pencil regime
  /// (`CertificateRegime::ComplexSymmetricPencil`) only; quiet NaN or false
  /// otherwise. The pairing is the complex bilinear restriction
  /// `B_C = (Phi^vee)^T G^U Phi` of the band. Its determinant and condition
  /// number are reported and no sign or inertia is extracted from it.
  /// `isotropic` marks `det B_C = 0`, the exceptional-point indicator, where
  /// the canonical left frame is refused with `leftFrameRefusal` naming the
  /// reason. `metricSymmetryDefect` is the regime's verification residual,
  /// `M L = (M L)^T`.
  std::complex<double> pairingDeterminant{std::numeric_limits<double>::quiet_NaN(),
                                          std::numeric_limits<double>::quiet_NaN()};
  double pairingCondition = std::numeric_limits<double>::quiet_NaN();
  double pairingScale = std::numeric_limits<double>::quiet_NaN();
  bool isotropic = false;
  std::string leftFrameRefusal{};
  double metricSymmetryDefect = std::numeric_limits<double>::quiet_NaN();
  /// Whether the fiber's stored left frame is the transpose dual
  /// \f$ \tilde\Phi \f$ itself (the chain-level pencil path, whichever
  /// regime its verification reached), rather than Psi normalized by
  /// Psi^dagger W Phi = I. `SpectralFiber::dualFrame` reads it.
  bool bilinearLeftFrame = false;
  /// The band's frequency window [min Re(lambda), max Re(lambda)], the window
  /// handed to the response API (see :class:`SpectralBandWindow`).
  double frequencyLower = std::numeric_limits<double>::quiet_NaN();
  double frequencyUpper = std::numeric_limits<double>::quiet_NaN();
  /// Whether the band was produced by a verified self-adjoint solve, i.e. the
  /// positive regime with Hermiticity or symmetry checked before the solver was
  /// applied. A self-adjoint solver is never applied to a non-self-adjoint
  /// operator.
  bool selfAdjoint = false;
  /// Whether the band met every certification threshold of the producing
  /// `SpectralFiberConfig`: isolation on `nearestDiscardedSeparation`,
  /// localization, residuals, Gram defect and projector conditioning. When
  /// false the band is still reported, as an uncertified read.
  bool accepted = false;
  /// The graded claim: domain `BandWindow`, regime as verified, grade
  /// `CertifiedNumerical` — or `AlgebraicallyExact` where the arithmetic is
  /// closed-form — when accepted. An uncertified band carries
  /// `HeuristicDiscovery`, which never `holds()`.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary: rank, window, gaps, signature and
  /// residuals. A degenerate rank is reported as an uninterpreted
  /// multiplicity.
  [[nodiscard]] std::string describe() const;
};

/// Principal-angle and support comparison of two fibers: the tracking
/// primitive. Angles are computed between the two frames' column spans
/// restricted to the shared cells, with cells matched by their sorted vertex-id
/// tuples rather than by index. Only the projector ranges enter, so the read is
/// both gauge- and relabeling-invariant.
struct FiberOverlapRead {
  /// Jaccard overlap of the two fibers' k-cell supports.
  double supportOverlap = 0.0;
  /// Number of shared k-cells.
  std::size_t sharedCells = 0;
  /// Principal angles (radians, ascending) between the two subspaces
  /// restricted to the shared cells.
  std::vector<double> principalAngles{};
  /// (sum_i cos^2 theta_i) / max(rank_a, rank_b) in [0, 1]; 1 exactly when
  /// the ranks agree and the restricted subspaces coincide.
  double subspaceOverlap = 0.0;
};

/// # SpectralFiber
///
/// One isolated spectral band of a component-restricted Hodge Laplacian: the
/// right and left frames, the band projector, the eigenvalues, and the
/// :class:`SpectralBandCertificate`.
///
/// The band is represented by its projector `P = Phi Phi~^T`, with `Phi~` the
/// algebraic (transpose) dual of the right frame, `Phi~^T Phi = I`
/// (`dualFrame()`): the one pairing of the complex-bilinear formulation, the
/// same in every regime. The solvers normalize their own left frame: on the
/// chain-level pencil path it is `Phi~` itself; elsewhere it is `Psi` with
/// `Psi^dagger W Phi = I`, and `Phi~ = W conj(Psi)`, so `P = Phi Psi^dagger W`.
/// Individual eigenvectors are a gauge choice and do not determine an identity
/// or a downstream observable. On the self-adjoint path `Psi = Phi` (a
/// W-orthonormal frame, `J = I`); in the Krein-normalizable signed regime
/// `Psi = Phi J` with `Phi^dagger W Phi = J = diag(I_p, -I_q)`; on the
/// biorthogonal, non-normal path `Phi` and `Psi` are matched right and left
/// subspace bases.
///
/// Instances are immutable value objects produced by
/// :class:`SpectralFiberTracker` (or rehydrated by `fromRecord`).
class SpectralFiber {
  public:
    SpectralFiber() = default;

    /// Assemble a fiber from its parts. `cells` are the k-cell sorted
    /// vertex-id tuples in the row order of the frames; `weights` is the
    /// diagonal W restricted to those cells.
    SpectralFiber(std::vector<std::vector<std::uint64_t>> cells,
                  std::vector<std::complex<double>> eigenvalues,
                  Eigen::MatrixXcd rightFrame, Eigen::MatrixXcd leftFrame,
                  Eigen::VectorXcd weights, SpectralBandCertificate certificate);

    /// Form degree of the certificate.
    [[nodiscard]] int degree() const noexcept { return certificate_.degree; }
    /// Band rank (columns of the frames).
    [[nodiscard]] std::size_t rank() const noexcept { return certificate_.rank; }
    /// Whether the certificate accepted the band.
    [[nodiscard]] bool accepted() const noexcept { return certificate_.accepted; }

    /// Right frame Phi (cells x rank).
    [[nodiscard]] Eigen::MatrixXcd rightFrame() const { return right_; }
    /// Left frame as produced by the regime's solver (cells x rank): Psi
    /// with Psi^dagger W Phi = I on the self-adjoint, Krein and non-normal
    /// paths; the canonical bilinear left frame Phi~ itself on the chain-level
    /// pencil path. `dualFrame()` is the one pairing across regimes.
    [[nodiscard]] Eigen::MatrixXcd leftFrame() const { return left_; }
    /// The algebraic (transpose) dual of the right frame, cells x rank:
    /// \f$ \tilde\Phi \f$ with \f$ \tilde\Phi^T \Phi = I \f$ — the pairing
    /// of the complex-bilinear formulation, the same in every regime. On the
    /// pencil path it is the stored left frame; on the other paths it is
    /// \f$ W \bar\Psi \f$, since \f$ (W\bar\Psi)^T \Phi = \Psi^\dagger W
    /// \Phi \f$. A refused left frame (isotropic band) reads as zero columns.
    /// `quantum::CovarianceState::fromBiorthogonalFrames(rightFrame(),
    /// dualFrame())` is the band's biorthogonal Slater covariance.
    [[nodiscard]] Eigen::MatrixXcd dualFrame() const;
    /// The band projector \f$ P = \Phi \tilde\Phi^T \f$ (cells x cells) in
    /// the transpose pairing, assembled on demand from the stored frames; it
    /// equals \f$ \Phi \Psi^\dagger W \f$ off the pencil path.
    [[nodiscard]] Eigen::MatrixXcd projector() const;
    /// The diagonal inner-product weights W restricted to the band's cells:
    /// the metric the Gram and signature certificates are measured in.
    [[nodiscard]] Eigen::VectorXcd weightDiagonal() const { return weights_; }

    /// The band's eigenvalues (with multiplicity), sorted by (Re, Im).
    [[nodiscard]] const std::vector<std::complex<double>> &eigenvalues() const noexcept {
      return eigenvalues_;
    }
    /// Mean of the band eigenvalues.
    [[nodiscard]] std::complex<double> bandCenter() const;

    /// The k-cells carrying the band, as sorted vertex-id tuples in frame row
    /// order; a single-vertex tuple per row at degree 0.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &cellVertices() const noexcept {
      return cells_;
    }

    /// The band certificate.
    [[nodiscard]] const SpectralBandCertificate &certificate() const noexcept {
      return certificate_;
    }

    /// Principal-angle and support comparison against another fiber (see
    /// :class:`FiberOverlapRead`). Cells are matched by vertex-id tuple.
    [[nodiscard]] static FiberOverlapRead overlap(const SpectralFiber &a,
                                                  const SpectralFiber &b);

    /// Checkpoint serialization: the schema-versioned JSON-able
    /// :class:`Record` of the fiber, with complex leaves split into
    /// `{name}_re` and `{name}_im`.
    [[nodiscard]] Record toRecord() const;
    /// Rehydrate a fiber from `toRecord()` output.
    /// @throws std::invalid_argument on an unknown `schema_version`.
    [[nodiscard]] static SpectralFiber fromRecord(const Record &record);

  private:
    std::vector<std::vector<std::uint64_t>> cells_{};
    std::vector<std::complex<double>> eigenvalues_{};
    Eigen::MatrixXcd right_{};
    Eigen::MatrixXcd left_{};
    Eigen::VectorXcd weights_{};
    SpectralBandCertificate certificate_{};
};

/// An accepted band's frequency window as plain data for the response
/// consumer: the lower and upper frequency bounds plus the band certificate. It
/// carries no operator, no frame, and no reference to any response type.
struct SpectralBandWindow {
  int degree = 0;
  std::size_t rank = 0;
  double frequencyLower = std::numeric_limits<double>::quiet_NaN();
  double frequencyUpper = std::numeric_limits<double>::quiet_NaN();
  SpectralBandCertificate certificate{};
};

/// One matched fiber pair across frames or resolutions.
struct FiberMatchRead {
  /// Positions in the `from` / `to` fiber lists handed to `matchFibers`.
  std::size_t fromIndex = 0;
  std::size_t toIndex = 0;
  int degree = 0;
  FiberOverlapRead overlap{};
  bool ranksEqual = false;
  /// Certified continuation: both endpoint bands accepted, equal rank, and
  /// subspace overlap at least the configured threshold. When an endpoint band's
  /// gap closed and it is uncertified, the match is reported but not certified,
  /// so identity never flips discontinuously.
  bool certifiedContinuation = false;
};

/// The band enumeration of one (component, degree) pair.
struct ComponentBandRead {
  /// The component's level-0 cell ids (vertex identifiers): the
  /// `ComponentRead::support` convention and the `AnalyticCache` component
  /// key.
  std::vector<std::uint64_t> support{};
  int degree = 0;
  /// Number of k-cells of the restricted operator (its dimension).
  std::size_t dimension = 0;
  /// The restricted operator's k-cells as sorted vertex-id tuples, in the
  /// canonical ChainComplex row and column order.
  std::vector<std::vector<std::uint64_t>> cellVertices{};
  /// The verified metric regime of the solve.
  cobordism::CertificateRegime regime =
      cobordism::CertificateRegime::NonNormal;
  /// Which solver ran: "dense-self-adjoint", "sparse-block-self-adjoint",
  /// or "dense-general".
  std::string solverPath{};
  /// Whether the sparse path covered only the lowest part of the spectrum.
  bool truncated = false;
  /// The computed (covered) eigenvalues, sorted by (Re, Im).
  std::vector<std::complex<double>> coveredEigenvalues{};
  /// The enumerated bands; every band is reported, certified or not.
  std::vector<SpectralFiber> fibers{};
  /// The eigensolve's own certificate: regime, residual and conditioning, plus
  /// the dense-reference deviation when cross-validation ran.
  cobordism::Certificate solveCertificate{};

  /// Checkpoint serialization of the whole read (fibers included).
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static ComponentBandRead fromRecord(const Record &record);
};

/// # SpectralFiberTracker
///
/// Extraction and tracking of isolated localized Hodge Laplacian bands on
/// persistent components.
///
/// Reference: Lim, "Hodge Laplacians on graphs", arXiv:1507.05379.
///
/// ## The restricted operator
///
/// For a component support `S` (vertex ids) the tracker assembles the weighted
/// Hodge Laplacian of the full induced subcomplex on `S` — every simplex all of
/// whose vertices lie in `S` — in the canonical ChainComplex cell order, with
/// the same diagonal inner-product weights \f$ W_k \f$ the whole-complex
/// `HodgeLaplacian` uses, including the degenerate-cell +1 fallback and the
/// `WeightConvention`:
///
/// \f[
///   L_k^S = (W_k^S)^{-1} (d_k^S)^T W_{k-1}^S d_k^S
///           + d_{k+1}^S (W_{k+1}^S)^{-1} (d_{k+1}^S)^T W_k^S ,
/// \f]
///
/// where \f$ d^S \f$ restricts the integer boundary maps to the cells inside
/// `S`. When `S` is the whole vertex set this is the whole-complex operator: on
/// the signed and complex-weight paths it equals `HodgeLaplacian::laplacian(k)`
/// entry for entry, pinned by the spectral resolution
/// \f$ \sum_{\mathrm{bands}} \Phi \Lambda \Psi^\dagger W = L \f$. On the
/// verified positive path the solved object is the symmetric W-orthonormal
/// similarity \f$ B_k^T B_k + B_{k+1} B_{k+1}^T \f$ with the same spectrum,
/// and the frames are mapped back to cochain coordinates
/// (\f$ \Phi = W^{-1/2} U \f$), so the same resolution identity holds.
///
/// At degree 0 the operator is the induced-subgraph Hermitian U(1) connection
/// graph Laplacian under `HodgeLaplacian::connectionLaplacian`'s conventions
/// (\f$ A_{ij} = \sum \ell^2 e^{i\varphi} \f$, magnitude degree), not the
/// Hodge `laplacian(0)`, which is \f$ d_1 W_1^{-1} d_1^T \f$ and a different
/// operator. The band structure a degree-0 fiber tracks is the Aharonov-Bohm
/// one, which only the connection operator carries. A `(k+1)`-cell with a
/// vertex outside `S` does not contribute: the component is read as a complex
/// in its own right.
///
/// ## Regimes
///
/// The regime is verified, never assumed.
///
///  - Positive: all participating weights real positive, and the degree-0
///    operator Hermitian by measurement. The symmetric representation
///    \f$ B_k^T B_k + B_{k+1} B_{k+1}^T \f$ is solved self-adjointly, by dense
///    solve below `denseCrossover` and by a deterministic sparse block
///    shift-invert subspace solve at and above it.
///  - Hermitian signed (Krein): weights real with negative entries, and the
///    operator W-self-adjoint (\f$ WL = (WL)^T \f$, verified). A general dense
///    solve supplies the band, and the certificate records the Krein inertia of
///    \f$ \Phi^\dagger W \Phi \f$ normalized to \f$ \mathrm{diag}(I_p,
///    -I_q) \f$.
///  - Non-normal: complex weights, or a failed self-adjointness verification.
///    Matched right and left subspaces with \f$ \Psi^\dagger W \Phi = I \f$
///    and the biorthogonal Riesz projector; both residuals and the band
///    conditioning are reported.
///
/// ## Band rule
///
/// Eigenvalues sorted by (Re, Im) are grouped into bands by the relative gap
/// rule of `SpectralFiberConfig`, and every band is reported with its projector
/// and certificate. Certification requires isolation from the nearest discarded
/// eigenvalue in the complex plane — sorting supplies the grouping, never the
/// isolation measurement — plus localization capped by `maxLocalizationExcess`,
/// residuals, Gram defect and projector conditioning. A closing gap yields an
/// uncertified band, not a different identity. The detector enumerates whatever
/// ranks appear; it never requests a particular rank, and no eigenvalue
/// threshold is used as a Betti-number oracle.
///
/// ## Read-only
///
/// Never calls a solver on the spacetime and never mutates it; nothing here
/// enters any emergence objective.
class SpectralFiberTracker {
  public:
    /// `AnalyticCache` kind string of the per-(component, degree) payload.
    static constexpr const char *kCacheKind = "spectral-fiber";

    /// Bind to the spacetime to read (kept alive by the `shared_ptr`), a
    /// configuration, and the Hodge weight convention, defaulting to
    /// `HodgeLaplacian::defaultWeightConvention()`.
    explicit SpectralFiberTracker(
        std::shared_ptr<Spacetime> st, SpectralFiberConfig cfg = {},
        cobordism::HodgeLaplacian::WeightConvention weights =
            cobordism::HodgeLaplacian::defaultWeightConvention());

    /// Bind with an explicit metric source. Under
    /// `HodgeLaplacian::MetricSource::WhitneyPencil` every degree
    /// \f$ k \ge 1 \f$ is read on the chain-level Whitney pencil of the
    /// induced subcomplex (`chainhodge::CovariantChainHodge`): the operator is
    /// \f$ h_k(s,U) \f$, the regime is the verified `ComplexSymmetricPencil`
    /// (or `NonNormal` when the transpose identity fails), and every band is
    /// the Riesz projector of a circular contour drawn around the gap-rule
    /// group, with right frame `Phi`, canonical left frame `Phi~`, and the
    /// bilinear pairing certificates `pairingDeterminant`, `pairingCondition`,
    /// `pairingScale` and `isotropic`. Degree 0 keeps the U(1) connection
    /// operator under either source.
    SpectralFiberTracker(std::shared_ptr<Spacetime> st, SpectralFiberConfig cfg,
                         cobordism::HodgeLaplacian::MetricSource source);

    [[nodiscard]] cobordism::HodgeLaplacian::MetricSource metricSource() const noexcept {
      return metricSource_;
    }

    [[nodiscard]] const SpectralFiberConfig &config() const noexcept {
      return cfg_;
    }
    [[nodiscard]] cobordism::HodgeLaplacian::WeightConvention
    weightConvention() const noexcept {
      return weights_;
    }

    /// Enumerate the bands of one component at one form degree. `support` is
    /// the component's vertex-id set; input order is irrelevant and unknown ids
    /// are ignored. An empty restricted operator yields a read with
    /// `dimension == 0` and no fibers.
    /// @throws std::invalid_argument for a negative degree.
    [[nodiscard]] ComponentBandRead enumerateBands(
        const std::vector<std::uint64_t> &support, int degree) const;

    /// Enumerate every configured degree on every component of a discovery
    /// result, in (component, degree) order.
    [[nodiscard]] std::vector<ComponentBandRead> enumerateOnComponents(
        const std::vector<ComponentRead> &components) const;

    /// `enumerateBands` through the `AnalyticCache` contract: served from the
    /// cache while the component's star is untouched, and otherwise recomputed
    /// and re-stored under the current revision. The cache must be bound to the
    /// same spacetime. Payload kind `kCacheKind`, parameter the degree.
    [[nodiscard]] ComponentBandRead enumerateBandsCached(
        cobordism::AnalyticCache &cache,
        const std::vector<std::uint64_t> &support, int degree) const;

    /// The accepted bands' frequency windows as plain data for the response
    /// consumer.
    [[nodiscard]] static std::vector<SpectralBandWindow> acceptedWindows(
        const std::vector<ComponentBandRead> &reads);

    /// Track fibers across frames or resolutions: for every `from` fiber, the
    /// best `to` partner of the same degree by subspace overlap (principal
    /// angles on shared cells), with ties broken by support overlap then
    /// position. Fibers with no positive-overlap partner are omitted.
    /// `certifiedContinuation` additionally requires both endpoint bands
    /// accepted, equal ranks, and overlap at least `overlapThreshold`.
    [[nodiscard]] static std::vector<FiberMatchRead> matchFibers(
        const std::vector<SpectralFiber> &from,
        const std::vector<SpectralFiber> &to, double overlapThreshold = 0.5);

  private:
    struct RestrictedOperator;  // assembled restricted Hodge data
    struct SolveOutput;         // one solve path's eigen-paired output

    std::shared_ptr<Spacetime> st_{};
    SpectralFiberConfig cfg_{};
    cobordism::HodgeLaplacian::WeightConvention weights_{
        cobordism::HodgeLaplacian::WeightConvention::SquaredContent};
    cobordism::HodgeLaplacian::MetricSource metricSource_{
        cobordism::HodgeLaplacian::MetricSource::DiagonalWeights};

    [[nodiscard]] RestrictedOperator assembleRestricted(
        const std::vector<std::uint64_t> &support, int degree) const;

    // The three solve paths.
    void solveDenseSelfAdjoint(const RestrictedOperator &op,
                               ComponentBandRead &read) const;
    void solveSparseSelfAdjoint(const RestrictedOperator &op,
                                ComponentBandRead &read) const;
    // The chain-level pencil path: Riesz bands on circular contours.
    void solvePencilBands(const RestrictedOperator &op,
                          ComponentBandRead &read) const;
    void solveDenseGeneral(const RestrictedOperator &op,
                           ComponentBandRead &read) const;

    // Band grouping plus per-band measurement and certification.
    void buildFibers(const RestrictedOperator &op, const SolveOutput &out,
                     ComponentBandRead &read) const;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SPECTRALFIBER_H
