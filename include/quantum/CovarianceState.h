// The quasi-free covariance layer.
//
// ─── What lives here ─────────────────────────────────────────────────────
//
//   • CovarianceState      — the number-conserving quasi-free state stored
//                            as its covariance matrix Γ_ij = ⟨a_j† a_i⟩:
//                            initialization from accepted band projectors
//                            (Γ = P), boundary-register occupations, or an
//                            occupied Slater frame; propagation by exact
//                            one-particle conjugation (both entry points: a
//                            Hermitian generator h and a supplied
//                            one-particle transport U); Wick contraction of
//                            every polynomial certificate; the mean-field
//                            self-consistency loop h = h(Γ, ·) with a
//                            purity/Gaussianity certificate per iteration;
//                            AnalyticCache-backed Wick reads; and
//                            checkpoint serialization of Γ.
//   • WickCertificateRead  — one Wick-evaluated certificate: value,
//                            measured residual, polynomialId,
//                            covarianceHash, Certificate.
//   • MeanFieldStepRead    — the per-iteration purity/Gaussianity
//                            certificate record of the mean-field loop.
//
// ─── Exact identities implemented (tested against dense Fock references) ─
//
//   Domain: every identity below is exact on a number-conserving quasi-free
//   state with covariance Γ (pure Slater or mixed Gaussian); "exact" means a
//   finite closed-form sum whose only error is double rounding.
//
//   • Wick determinant:  ⟨a†_{i1}···a†_{ip} a_{jp}···a_{j1}⟩
//       = det[ Γ_{j_l i_k} ]_{k,l}  (creators/annihilators in paired slot
//     order; duplicate creators give a repeated row, hence exactly zero —
//     Pauli exclusion is the determinant's alternation).
//   • Smeared Gram/Pauli determinant:
//       ⟨a†(v_1)···a†(v_p) a(w_p)···a(w_1)⟩ = det( W† Γ V )
//     with the ExteriorAlgebra smearing conventions (a†(v) linear, a(w)
//     antilinear).
//   • Parity: ⟨(−1)^N⟩ = det(I − 2Γ); subset parity ⟨(−1)^{N_S}⟩ =
//     det(I_S − 2Γ_S) on the principal submatrix. (In a Nambu/pairing
//     extension this read becomes a Pfaffian; name and shape are stable.)
//   • Color wedge: |S_ABC|² = ⟨a†(c_1)a†(c_2)a†(c_3) a(c_3)a(c_2)a(c_1)⟩
//       = det( C† Γ C );  when Γ is the Slater projector onto colspan(C)
//     this equals det(C†C) = |det C|² — exactly ColorFiber::singletGram /
//     |ColorFiber::colorWedge|², the cross-checked certificate.
//   • Ordered bilinear moments (the quartic and octic Wick sums): from the
//     multiplicative second quantization ⟨Λ•(M)⟩ = det(I + (M − I)Γ),
//       ⟨dΓ(A_1)···dΓ(A_n)⟩ = [s_1···s_n] det(I + (∏_k(I + s_kA_k) − I)Γ)
//     evaluated exactly by the set-partition/ordered-composition trace
//     expansion (no 2^M object, no term-by-term index sums). Closed n = 2
//     form: tr(A B Γ) − tr(A Γ B Γ) + tr(A Γ)tr(B Γ).
//   • Total spin: ⟨J²⟩ = Σ_α ⟨dΓ(J_α)dΓ(J_α)⟩ (quartic Wick sums) and
//       Var(J²) = Σ_{αβ} ⟨dΓ(J_α)dΓ(J_α)dΓ(J_β)dΓ(J_β)⟩ − ⟨J²⟩²
//     (octic Wick sums), with caller-supplied one-particle spin matrices.
//   • Propagation: Γ(t) = e^{−iht} Γ e^{+iht} is the exact solution of
//       iΓ̇ = [h, Γ];  conjugation by a unitary preserves Hermiticity, the
//     spectrum, and Γ² = Γ exactly — quadratic evolution (and the declared
//     mean-field self-consistency h = h(Γ, ·)) never leaves the Gaussian
//     manifold. Purity is a measured certificate ‖Γ² − Γ‖_F, never an
//     assumption; mixed quasi-free states report the covariance-spectrum
//     constraint dist(spec Γ, [0,1]) instead.
//
// ─── No Fock vector on the quasi-free path ───────────────────────────────
//
// The production representation is the M×M covariance: every method here is
// polynomial in the mode count (dense M×M linear algebra) and no code path
// allocates a 2^M object. Dense Fock constructions (ExteriorAlgebra,
// independent Jordan-Wigner chains) are test references; LazyFockEngine is
// the oracle layer and the carrier for explicitly non-Gaussian boundary
// data. The mean-field loop takes h from the caller.
//
// ─── Emergence sub-modes served ──────────────────────────────────────────
//
//   • strict: the carried state does not act back on the geometry — plain
//     evolve()/applyTransport() with a Γ-independent generator.
//   • certificates_blind_mean_field: the generator may depend on Γ (and on
//     classical geometry closed over by the caller) via meanFieldEvolve();
//     the loop reads no particle certificate, and every iteration carries a
//     purity/Gaussianity certificate. Backreaction is not evidence of a
//     genuine non-Gaussian interaction.

#pragma once

#include <Eigen/Dense>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "cobordism/Certificate.h"
#include "observables/Record.h"

namespace tessera::cobordism {
class AnalyticCache;
}

namespace tessera::quantum {

/// One Wick-evaluated polynomial certificate: the value,
/// the measured residual, the normal-ordered observable / contraction-plan
/// identifier, the covariance fingerprint the value was read from, and the
/// certification record grading the claim.
struct WickCertificateRead {
    /// The Wick-contracted expectation value. Real-valued observables
    /// (occupations, parities, |S_ABC|², ⟨J²⟩, Var(J²)) report their
    /// imaginary rounding leakage in `residual` — nothing is silently
    /// `.real()`-ed away.
    std::complex<double> value{0.0, 0.0};
    /// Measured rounding/premise residual: the covariance Hermiticity defect,
    /// maximized with the imaginary leakage |Im value| for observables that
    /// are real by construction.
    double residual{0.0};
    /// Identifies the normal-ordered observable and contraction plan (e.g.
    /// "parity", "occupation[3]", "spin-squared-variance[fp]"); matrix-
    /// parametrized reads embed a content fingerprint of their coefficient
    /// matrices so distinct observables never share an identifier.
    std::string polynomialId{};
    /// Fingerprint of the covariance the value was evaluated on
    /// (CovarianceState::covarianceHash) — a cached read is served only for
    /// a matching hash.
    std::string covarianceHash{};
    /// The graded claim: AlgebraicallyExact / Static, with the regime
    /// verified on the covariance (PositiveSemidefinite when Γ is Hermitian
    /// with spectrum in [0,1] within tolerance, HermitianIndefinite when
    /// Hermitian with out-of-range spectrum, NonNormal otherwise).
    cobordism::Certificate certificate{};
};

/// The per-iteration certificate record of the mean-field self-consistency
/// loop: the measured defects after the step and the purity/Gaussianity
/// certificate.
struct MeanFieldStepRead {
    /// Iteration index (0-based) and the accumulated evolution time.
    std::size_t step{0};
    double time{0.0};
    /// Relative Hermiticity defect of the caller-supplied generator h for
    /// this step, ‖h − h†‖_F / max(1, ‖h‖_F).
    double generatorHermiticityDefect{0.0};
    /// Covariance defects measured after the step: Hermiticity
    /// ‖Γ − Γ†‖_F / max(1, ‖Γ‖_F), purity ‖Γ² − Γ‖_F (the pure-Slater
    /// certificate), and the covariance-spectrum constraint
    /// max_i dist(λ_i(Γ), [0, 1]) (the mixed-state certificate).
    double hermiticityDefect{0.0};
    double purityDefect{0.0};
    double occupationSpectrumDefect{0.0};
    /// The Gaussianity certificate of this iteration: AlgebraicallyExact /
    /// Static (the conjugation is closed-form); residual = the pure-path
    /// purity defect when the loop entered on a pure state (purity defect
    /// within tolerance), else the mixed-path spectrum constraint — each
    /// maximized with the Hermiticity defects above.
    cobordism::Certificate certificate{};
};

/// # CovarianceState
///
/// The number-conserving quasi-free state represented exactly by its
/// covariance matrix \f$ \Gamma_{ij} = \langle a_j^\dagger a_i\rangle \f$ —
/// the production state path of the quasi-free sector. Pure Slater states
/// satisfy \f$ \Gamma^2 = \Gamma \f$ — reported as the
/// measured `purityDefect()`, never assumed. Every polynomial observable is
/// evaluated by Wick contraction as a finite exact sum (see the header
/// identity list); dense Fock references live in tests and in
/// LazyFockEngine, never on this path.
///
/// ## Nambu/pairing shape (implemented: number-conserving)
///
/// The API is shaped for a pairing sector:
/// `numberConserving()` reports the sector, `pairing()` is the anomalous
/// block \f$ F_{ij} = \langle a_j a_i \rangle \f$ (identically zero here),
/// and `nambuCovariance()` returns the full doubled covariance
/// \f$ G = \begin{pmatrix} \Gamma & F \\ -\bar F & I - \Gamma^T
/// \end{pmatrix} \f$ (number-conserving: block-diagonal), which is idempotent
/// exactly when Γ is. With F populated, parity-style determinant reads
/// become Pfaffians.
///
/// ## Mutability and certificates
///
/// `evolve` / `applyTransport` / `meanFieldEvolve` mutate Γ; every defect
/// (`hermiticityDefect`, `purityDefect`, `occupationSpectrumDefect`) is
/// measured on demand and cached per Γ revision. Wick reads carry a
/// `WickCertificateRead` with the verified regime; `covarianceHash()`
/// fingerprints the exact double bit patterns of Γ, so replay and cache
/// consistency are byte-exact statements.
class CovarianceState {
  public:
    using Complex = std::complex<double>;

    /// `AnalyticCache` kind string of cached Wick reads.
    static constexpr const char* kCacheKind = "wick-read";

    /// Largest number of bilinear factors `wickBilinearMoment` accepts (the
    /// ordered-composition expansion is exact but combinatorial in n, and
    /// polynomial in M for fixed n; the observables here need n ≤ 4).
    static constexpr std::size_t kMaxBilinearFactors = 8;

    /// Adopt an explicit covariance matrix (boundary/covariance data as
    /// plain numbers). No symmetrization, no clamping: defects are measured,
    /// reported, and graded — never repaired.
    /// @throws std::invalid_argument when `gamma` is not square.
    explicit CovarianceState(Eigen::MatrixXcd gamma);

    /// Γ = P from an accepted band projector
    /// (`SpectralFiber::projector()` output consumed as a plain matrix). On
    /// the self-adjoint path P is an orthogonal projector, hence a pure
    /// Slater covariance; an oblique (Krein/biorthogonal) projector is
    /// adopted verbatim and its Hermiticity defect is reported.
    [[nodiscard]] static CovarianceState fromBandProjector(
        const Eigen::MatrixXcd& projector);

    /// Diagonal covariance Γ = diag(n) from boundary-register occupation
    /// data, n_i = ⟨n_i⟩ ∈ [0, 1] (values outside [0, 1] are adopted and
    /// show up in `occupationSpectrumDefect`).
    [[nodiscard]] static CovarianceState fromOccupations(
        const Eigen::VectorXd& occupations);

    /// Pure Slater covariance Γ = Φ (Φ†Φ)^{-1} Φ† from an M×N frame of
    /// occupied one-particle orbitals (boundary-register state vectors enter
    /// here as occupied columns). The frame need not be orthonormal — Γ is
    /// the exact orthogonal projector onto its column span.
    /// @throws std::invalid_argument when the frame is empty of rows or
    ///         rank-deficient (relative pivot below `rankTolerance`).
    [[nodiscard]] static CovarianceState fromSlaterFrame(
        const Eigen::MatrixXcd& orbitals, double rankTolerance = 1e-12);

    // ── state data ───────────────────────────────────────────────────────

    /// Mode count M (Γ is M×M).
    [[nodiscard]] std::size_t modeCount() const noexcept {
        return static_cast<std::size_t>(gamma_.rows());
    }

    /// The covariance matrix Γ, \f$ \Gamma_{ij} = \langle a_j^\dagger a_i
    /// \rangle \f$.
    [[nodiscard]] const Eigen::MatrixXcd& gamma() const noexcept {
        return gamma_;
    }

    /// Whether this state lives in the number-conserving sector (always
    /// true in this implementation; a pairing extension reports false and
    /// populates `pairing()`).
    [[nodiscard]] bool numberConserving() const noexcept { return true; }

    /// The anomalous pairing block \f$ F_{ij} = \langle a_j a_i \rangle \f$
    /// — identically zero in the number-conserving sector (returned
    /// explicitly so consumers are Nambu-shaped today).
    [[nodiscard]] Eigen::MatrixXcd pairing() const;

    /// The full 2M×2M Nambu covariance over the doubled generators
    /// α = (a_1..a_M, a_1†..a_M†), G_{kl} = ⟨α_l‡ α_k⟩: blocks
    /// [[Γ, F], [−F̄, I − Γᵀ]] (number-conserving: F = 0). Idempotent
    /// exactly when Γ is.
    [[nodiscard]] Eigen::MatrixXcd nambuCovariance() const;

    /// ⟨n_i⟩ = Γ_ii of one mode (the raw complex diagonal entry — real up
    /// to the Hermiticity defect). @throws std::invalid_argument when
    /// `mode >= modeCount()`.
    [[nodiscard]] Complex occupation(std::size_t mode) const;

    /// The diagonal ⟨n_i⟩ for every mode.
    [[nodiscard]] Eigen::VectorXcd occupations() const;

    /// ⟨N⟩ = tr Γ.
    [[nodiscard]] Complex particleNumber() const;

    // ── measured defects and certificates ────────────────────────────────

    /// Relative Hermiticity defect ‖Γ − Γ†‖_F / max(1, ‖Γ‖_F).
    [[nodiscard]] double hermiticityDefect() const;

    /// Purity defect ε_purity = ‖Γ² − Γ‖_F — exactly zero for a pure Slater
    /// state. An O(1) value is a mixed (or invalid) covariance reporting
    /// itself, never an error.
    [[nodiscard]] double purityDefect() const;

    /// The mixed-state covariance-spectrum constraint
    /// max_i dist(λ_i((Γ+Γ†)/2), [0, 1]) — zero for every valid quasi-free
    /// covariance, pure or mixed.
    [[nodiscard]] double occupationSpectrumDefect() const;

    /// The purity certificate of the pure-Slater path:
    /// AlgebraicallyExact / Static with residual
    /// max(purityDefect, hermiticityDefect) against `tolerance`, in the
    /// verified regime. A mixed state does not hold() it.
    [[nodiscard]] cobordism::Certificate purityCertificate(
        double tolerance = 1e-9) const;

    /// Order-sensitive fingerprint of the exact double bit patterns of Γ
    /// (16 hex digits) — the `WickCertificateRead::covarianceHash` and the
    /// cached-read consistency key. A pure function of Γ: replay-stable.
    [[nodiscard]] std::string covarianceHash() const;

    // ── propagation (both entry points) ─────────────────────────────────

    /// Advance Γ by `dt` under the Hermitian one-particle generator h:
    /// Γ ← e^{−ih·dt} Γ e^{+ih·dt}, the exact solution of iΓ̇ = [h, Γ]
    /// (unitary conjugation through the eigendecomposition of h — no
    /// step-size error; preserves Hermiticity, spectrum, and purity to
    /// round-off).
    /// @throws std::invalid_argument on a shape mismatch or when h fails
    ///         Hermiticity: ‖h − h†‖_F > hermitianTolerance · max(1, ‖h‖_F).
    void evolve(const Eigen::MatrixXcd& h, double dt,
                double hermitianTolerance = 1e-9);

    /// The one-particle propagator e^{−ih·dt} `evolve` conjugates by —
    /// exposed so the two entry points can be pinned equal in tests:
    /// evolve(h, dt) ≡ applyTransport(propagator(h, dt)).
    /// @throws std::invalid_argument as `evolve`.
    [[nodiscard]] static Eigen::MatrixXcd propagator(
        const Eigen::MatrixXcd& h, double dt,
        double hermitianTolerance = 1e-9);

    /// Conjugate Γ by the one-particle transport of a cobordism step:
    /// Γ ← U Γ U†. A unitary U preserves Hermiticity, spectrum, and purity
    /// exactly; a leaky (non-unitary) transport's effect shows up in the
    /// defect reads afterwards and is never repaired.
    /// @throws std::invalid_argument on a shape mismatch.
    void applyTransport(const Eigen::MatrixXcd& transport);

    // ── mean-field self-consistency ─────────────────────────────────────

    /// The certificates-blind mean-field loop: for each of `steps`
    /// iterations, obtain h = `hamiltonian`(Γ) from the caller (classical
    /// geometry g is closed over by the caller), advance by `dt` via
    /// `evolve`, and record the per-iteration purity/Gaussianity
    /// certificate. Generalized Hartree-Fock dynamics: nonlinear in Γ but
    /// Gaussian-closed — the certificate measures that closure every step.
    /// The pure/mixed certificate path is chosen once, on entry: pure when
    /// purityDefect() ≤ `purityTolerance`.
    /// @throws std::invalid_argument when the callback returns a wrongly
    ///         shaped or non-Hermitian generator (as `evolve`).
    [[nodiscard]] std::vector<MeanFieldStepRead> meanFieldEvolve(
        const std::function<Eigen::MatrixXcd(const Eigen::MatrixXcd&)>&
            hamiltonian,
        double dt, std::size_t steps, double hermitianTolerance = 1e-9,
        double purityTolerance = 1e-9);

    // ── Wick reads (each a finite exact sum; see header identities) ─────

    /// ⟨n_mode⟩ as a certified read (value = Γ_mm).
    [[nodiscard]] WickCertificateRead wickOccupation(std::size_t mode) const;

    /// ⟨N⟩ = tr Γ as a certified read.
    [[nodiscard]] WickCertificateRead wickTotalNumber() const;

    /// Fermion parity ⟨(−1)^N⟩ = det(I − 2Γ).
    [[nodiscard]] WickCertificateRead wickParity() const;

    /// Subset parity ⟨(−1)^{N_S}⟩ = det(I_S − 2Γ_S) over the principal
    /// submatrix on `modes` (duplicates rejected).
    /// @throws std::invalid_argument on an out-of-range or duplicate mode.
    [[nodiscard]] WickCertificateRead wickSubsetParity(
        const std::vector<std::size_t>& modes) const;

    /// The Wick determinant of an elementary normal-ordered monomial in
    /// paired slot order:
    /// ⟨a†_{c_1}···a†_{c_p} a_{a_p}···a_{a_1}⟩ = det[Γ_{a_l c_k}]_{k,l}
    /// (annihilators applied in reversed list order, so `creators[k]` and
    /// `annihilators[k]` are matching slots; equal distinct lists give the
    /// joint occupation ⟨n_{c_1}···n_{c_p}⟩). Mismatched list lengths are
    /// exactly zero on a number-conserving state. p = 0 gives ⟨1⟩ = 1.
    /// @throws std::invalid_argument on an out-of-range mode.
    [[nodiscard]] WickCertificateRead wickNormalOrdered(
        const std::vector<std::size_t>& creators,
        const std::vector<std::size_t>& annihilators) const;

    /// The smeared Gram/Pauli determinant
    /// ⟨a†(v_1)···a†(v_p) a(w_p)···a(w_1)⟩ = det(W† Γ V) with the
    /// ExteriorAlgebra smearing conventions (columns of V create, columns of
    /// W annihilate).
    /// Mismatched column counts are exactly zero (number conservation).
    /// @throws std::invalid_argument when a frame's rows ≠ modeCount().
    [[nodiscard]] WickCertificateRead wickGramDeterminant(
        const Eigen::MatrixXcd& creatorFrame,
        const Eigen::MatrixXcd& annihilatorFrame) const;

    /// The color-wedge certificate |S_ABC|² = det(C† Γ C) of three color
    /// columns — the joint occupation weight of the color triad. When Γ is
    /// the Slater projector onto colspan(C) this is exactly
    /// ColorFiber::singletGram(C) = |ColorFiber::colorWedge(C)|², the
    /// cross-checked determinant certificate.
    /// @throws std::invalid_argument when C is not modeCount()×3.
    [[nodiscard]] WickCertificateRead wickColorWedgeSquared(
        const Eigen::MatrixXcd& colorColumns) const;

    /// The ordered bilinear moment ⟨dΓ(A_1)···dΓ(A_n)⟩ (operator order =
    /// list order), evaluated by the exact set-partition/ordered-composition
    /// trace expansion of det(I + (∏(I + s_kA_k) − I)Γ) — the general
    /// quartic (n = 2) and octic (n = 4) Wick engine. Cost: O(B(n)·n·M³)
    /// with B(n) the ordered-composition count — polynomial in M, no 2^M
    /// object.
    /// @throws std::invalid_argument on a shape mismatch, an empty list, or
    ///         n > kMaxBilinearFactors.
    [[nodiscard]] WickCertificateRead wickBilinearMoment(
        const std::vector<Eigen::MatrixXcd>& oneParticleFactors) const;

    /// ⟨J²⟩ = Σ_α ⟨dΓ(J_α)²⟩ from caller-supplied one-particle spin
    /// matrices (quartic Wick sums). Real up to rounding for Hermitian J
    /// and Γ; the imaginary leakage is the reported residual.
    /// @throws std::invalid_argument on a shape mismatch.
    [[nodiscard]] WickCertificateRead wickSpinSquaredExpectation(
        const Eigen::MatrixXcd& jx, const Eigen::MatrixXcd& jy,
        const Eigen::MatrixXcd& jz) const;

    /// Var(J²) = ⟨(J²)²⟩ − ⟨J²⟩² with ⟨(J²)²⟩ = Σ_{αβ}
    /// ⟨dΓ(J_α)dΓ(J_α)dΓ(J_β)dΓ(J_β)⟩ (octic Wick sums). Exactly zero on a
    /// J² eigenstate — a candidate with the right expectation and nonzero
    /// variance is not a certified sharp spin.
    /// @throws std::invalid_argument on a shape mismatch.
    [[nodiscard]] WickCertificateRead wickSpinSquaredVariance(
        const Eigen::MatrixXcd& jx, const Eigen::MatrixXcd& jy,
        const Eigen::MatrixXcd& jz) const;

    // ── cached Wick reads (the AnalyticCache contract) ──────────────────

    /// Fetch-or-compute one Wick read through the AnalyticCache: key =
    /// (component vertex set, kind = kCacheKind, parameter = a mixed
    /// fingerprint of `polynomialId` and the current covarianceHash). A hit
    /// is served only when the cache's geometry-freshness contract holds
    /// and the stored read's polynomialId and covarianceHash both match the
    /// request (a Γ change is a state change: verified explicitly, so a
    /// stale-Γ payload can only cause recomputation, never a wrong serve).
    /// On a miss, `compute` runs cold and the result is stored with its
    /// certificate.
    [[nodiscard]] WickCertificateRead wickReadCached(
        cobordism::AnalyticCache& cache,
        const std::vector<std::uint64_t>& componentVertexIds,
        const std::string& polynomialId,
        const std::function<WickCertificateRead()>& compute) const;

    // ── checkpoint serialization ────────────────────────────────────────

    /// The JSON-able checkpoint Record of Γ (schema-versioned; complex
    /// leaves split `{name}_re` / `{name}_im`; the measured defects are
    /// stored as informational channels and recomputed on load).
    [[nodiscard]] observables::Record toRecord() const;

    /// Rehydrate from `toRecord()` output. Rejects an unknown
    /// `schema_version` (std::invalid_argument), matching the checkpoint
    /// reader contract.
    [[nodiscard]] static CovarianceState fromRecord(
        const observables::Record& record);

  private:
    /// Verified-regime + residual assembly shared by every Wick read.
    [[nodiscard]] WickCertificateRead makeRead(Complex value,
                                               bool realByConstruction,
                                               std::string polynomialId) const;
    /// ⟨dΓ(A_1)···dΓ(A_n)⟩ core (validated inputs).
    [[nodiscard]] Complex bilinearMoment(
        const std::vector<Eigen::MatrixXcd>& factors) const;
    /// Content fingerprint of a matrix (for polynomialIds of
    /// matrix-parametrized reads).
    [[nodiscard]] static std::uint64_t matrixFingerprint(
        const Eigen::MatrixXcd& m, std::uint64_t seed);
    void invalidateDefects() noexcept;

    Eigen::MatrixXcd gamma_{};
    // Per-Γ-revision lazy defect caches (< 0 = not yet measured).
    mutable double hermiticityDefect_{-1.0};
    mutable double purityDefect_{-1.0};
    mutable double spectrumDefect_{-1.0};
};

}  // namespace tessera::quantum
