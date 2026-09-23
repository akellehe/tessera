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
// ─── Two duals: the Hermitian adjoint and the transpose ──────────────────
//
//   The whitepaper (Section 7) pairs a right Slater state with an algebraic
//   left dual, never with its Hermitian adjoint:
//       |Ξ_R⟩ = φ_1 ∧ ··· ∧ φ_N,   ⟨Ξ_L| = φ̃_1 ∧ ··· ∧ φ̃_N,   Φ̃ᵀΦ = I_N,
//   and the one-body datum is the biorthogonal Slater covariance
//       Γ = ΦΦ̃ᵀ,   Γ_ij = ⟨Ξ_L|a_j† a_i|Ξ_R⟩,   Γ² = Γ,   tr Γ = N,
//   an algebraic covariance, not a positive density matrix. Here ⟨Ξ_L| is the
//   linear functional ⟨0|ã(φ̃_N)···ã(φ̃_1) with ã(w) = Σ_i w_i a_i (the
//   contraction by the dual mode — linear, never conjugated), so
//   ⟨Ξ_L|Ξ_R⟩ = det(Φ̃ᵀΦ) = 1. The two frames evolve, for any complex
//   one-particle generator h (no h†, no scalar renormalization, no positive
//   metric), by
//       iΦ̇ = hΦ,   −iΦ̃̇ᵀ = Φ̃ᵀh,   hence   iΓ̇ = [h, Γ],
//   i.e. Φ(t) = e^{−iht}Φ and Φ̃(t) = e^{+ihᵀt}Φ̃; the dual pairing Φ̃ᵀΦ and
//   the idempotency Γ² = Γ are preserved exactly. The biorthogonal Wick
//   theorem makes every Γ-indexed read above (occupations, parities, the
//   Wick determinant, the bilinear moments) a transition amplitude of the
//   pair; the transpose-smeared Gram determinant
//       ⟨a†(v_1)···a†(v_p) ã(w_p)···ã(w_1)⟩ = det(Wᵀ Γ V)
//   is the pairing's own smeared read. `CovarianceDual::Transpose` states
//   carry both frames; `CovarianceDual::HermitianAdjoint` is the special case
//   of a certified ∗-structure (Φ̃ᵀ = Φ†), where h must be Hermitian and a
//   transport conjugates by U†.
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
#include <limits>
#include <string>
#include <vector>

#include "cobordism/Certificate.h"
#include "observables/Record.h"

namespace tessera::cobordism {
class AnalyticCache;
}

namespace tessera::quantum {

/// The dual a `CovarianceState` is paired against.
enum class CovarianceDual {
    /// Γ_ij = ⟨Ψ|a_j† a_i|Ψ⟩ with the Hermitian adjoint of one state (pure
    /// or mixed): the special case of a certified ∗-structure. The
    /// generator must be Hermitian and a transport conjugates Γ by U†.
    HermitianAdjoint,
    /// Γ = ΦΦ̃ᵀ over a matched right/left Slater pair (Φ̃ᵀΦ = I): the
    /// biorthogonal Slater covariance of the complex-bilinear formulation.
    /// Both frames are carried; any complex generator evolves them.
    Transpose,
};

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
    /// Measured rounding/premise residual. Hermitian-adjoint path: the
    /// covariance Hermiticity defect, maximized with the imaginary leakage
    /// |Im value| for observables that are real by construction. Transpose
    /// path: the dual-pairing defect ‖Φ̃ᵀΦ − I‖_F, the premise of the
    /// biorthogonal Wick theorem (transition amplitudes are complex, so no
    /// imaginary part is a defect there).
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
    /// Transpose path: the dual-pairing defect ‖Φ̃ᵀΦ − I‖_F after the step.
    /// Quiet NaN (unmeasured) on the Hermitian-adjoint path, which carries
    /// no frames.
    double dualityDefect{std::numeric_limits<double>::quiet_NaN()};
    /// The Gaussianity certificate of this iteration: AlgebraicallyExact /
    /// Static (the conjugation is closed-form). Hermitian-adjoint path:
    /// residual = the pure-path purity defect when the loop entered on a
    /// pure state (purity defect within tolerance), else the mixed-path
    /// spectrum constraint — each maximized with the Hermiticity defects
    /// above. Transpose path: residual = max(purity defect, duality defect);
    /// the Hermiticity defects are reported and are not defects there.
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
/// ## The two duals
///
/// A state built by `fromBiorthogonalFrames` is on the `Transpose` dual: it
/// carries a right frame Φ and its algebraic left dual Φ̃ with Γ = ΦΦ̃ᵀ,
/// evolves both frames under any complex generator (no h†), and reads
/// transition amplitudes of the pair. Every other constructor is on the
/// `HermitianAdjoint` dual — the special case of a certified ∗-structure —
/// where the generator must be Hermitian. For a Hermitian h the two paths
/// agree: the pair (Φ, Φ̄) of an orthonormal frame evolves to the same Γ.
///
/// ## Mutability and certificates
///
/// `evolve` / `applyTransport` / `meanFieldEvolve` mutate Γ; every defect
/// (`hermiticityDefect`, `purityDefect`, `occupationSpectrumDefect`,
/// `dualityDefect`) is measured on demand and cached per Γ revision. Wick
/// reads carry a
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

    /// The biorthogonal Slater covariance Γ = ΦΦ̃ᵀ of a matched right/left
    /// pair: `rightFrame` Φ (M×N, the occupied right modes of |Ξ_R⟩) and
    /// `leftFrame` Φ̃ (M×N, their algebraic duals in ⟨Ξ_L|), paired by the
    /// transpose, Φ̃ᵀΦ = I_N. The state is on the `Transpose` dual and
    /// carries both frames. The pairing is adopted verbatim: its defect
    /// ‖Φ̃ᵀΦ − I‖_F is measured (`dualityDefect()`) and reported, never
    /// repaired by a renormalization. N = 0 is the vacuum, Γ = 0.
    /// A `SpectralFiber`'s `rightFrame()` and `dualFrame()`, or a chain-level
    /// Riesz band's `frame` and `leftFrame`, enter here as they are.
    /// @throws std::invalid_argument when the frames have no mode rows or
    ///         their shapes differ.
    [[nodiscard]] static CovarianceState fromBiorthogonalFrames(
        const Eigen::MatrixXcd& rightFrame, const Eigen::MatrixXcd& leftFrame);

    // ── state data ───────────────────────────────────────────────────────

    /// Mode count M (Γ is M×M).
    [[nodiscard]] std::size_t modeCount() const noexcept {
        return static_cast<std::size_t>(gamma_.rows());
    }

    /// The dual the covariance is paired against (see `CovarianceDual`).
    [[nodiscard]] CovarianceDual dual() const noexcept { return dual_; }

    /// The right frame Φ (M×N) of a `Transpose` state; 0×0 on the
    /// Hermitian-adjoint path, which carries no frames.
    [[nodiscard]] const Eigen::MatrixXcd& rightFrame() const noexcept {
        return right_;
    }

    /// The left frame Φ̃ (M×N) of a `Transpose` state, Γ = ΦΦ̃ᵀ; 0×0 on the
    /// Hermitian-adjoint path.
    [[nodiscard]] const Eigen::MatrixXcd& leftFrame() const noexcept {
        return left_;
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

    /// The dual-pairing defect ‖Φ̃ᵀΦ − I_N‖_F of a `Transpose` state — the
    /// premise of the biorthogonal Wick theorem, preserved exactly by the
    /// two-frame evolution. Quiet NaN (unmeasured) on the Hermitian-adjoint
    /// path.
    [[nodiscard]] double dualityDefect() const;

    /// The purity certificate of the pure-Slater path:
    /// AlgebraicallyExact / Static with residual
    /// max(purityDefect, hermiticityDefect) against `tolerance` on the
    /// Hermitian-adjoint path and max(purityDefect, dualityDefect) on the
    /// `Transpose` path, in the verified regime. A mixed state does not
    /// hold() it.
    [[nodiscard]] cobordism::Certificate purityCertificate(
        double tolerance = 1e-9) const;

    /// Order-sensitive fingerprint of the exact double bit patterns of Γ
    /// (16 hex digits) — the `WickCertificateRead::covarianceHash` and the
    /// cached-read consistency key. A pure function of Γ: replay-stable.
    [[nodiscard]] std::string covarianceHash() const;

    // ── propagation (both entry points) ─────────────────────────────────

    /// Advance Γ by `dt` under the one-particle generator h, the exact
    /// solution Γ ← e^{−ih·dt} Γ e^{+ih·dt} of iΓ̇ = [h, Γ] (no step-size
    /// error).
    ///
    /// Hermitian-adjoint path: h must be Hermitian; unitary conjugation
    /// through the eigendecomposition of h preserves Hermiticity, spectrum,
    /// and purity to round-off.
    ///
    /// `Transpose` path: h is any complex matrix (non-Hermitian, e.g.
    /// complex-symmetric; defective h is fine) and no h† is formed. The two
    /// frames evolve by iΦ̇ = hΦ and −iΦ̃̇ᵀ = Φ̃ᵀh:
    /// Φ ← e^{−ih·dt}Φ and Φ̃ ← (e^{+ih·dt})ᵀΦ̃, each exponential taken
    /// directly (`complexPropagator`), and Γ ← ΦΦ̃ᵀ. The pairing Φ̃ᵀΦ and
    /// Γ² = Γ are preserved to round-off; `hermitianTolerance` is unused.
    /// @throws std::invalid_argument on a shape mismatch, or on the
    ///         Hermitian-adjoint path when h fails Hermiticity:
    ///         ‖h − h†‖_F > hermitianTolerance · max(1, ‖h‖_F).
    void evolve(const Eigen::MatrixXcd& h, double dt,
                double hermitianTolerance = 1e-9);

    /// The one-particle propagator e^{−ih·dt} of the Hermitian-adjoint path
    /// (eigendecomposition of a verified Hermitian h) — exposed so the two
    /// entry points can be pinned equal in tests:
    /// evolve(h, dt) ≡ applyTransport(propagator(h, dt)).
    /// @throws std::invalid_argument as `evolve` on the Hermitian path.
    [[nodiscard]] static Eigen::MatrixXcd propagator(
        const Eigen::MatrixXcd& h, double dt,
        double hermitianTolerance = 1e-9);

    /// The one-particle propagator e^{−ih·dt} of an arbitrary complex
    /// generator h (Padé scaling and squaring; no eigendecomposition, so
    /// a defective h is fine and no h† is formed). The `Transpose` path
    /// evolves Φ by complexPropagator(h, dt) and Φ̃ by
    /// complexPropagator(h, −dt)ᵀ.
    /// @throws std::invalid_argument when h is not square.
    [[nodiscard]] static Eigen::MatrixXcd complexPropagator(
        const Eigen::MatrixXcd& h, double dt);

    /// Carry the state through the one-particle transport U of a cobordism
    /// step.
    ///
    /// Hermitian-adjoint path: Γ ← U Γ U†. A unitary U preserves
    /// Hermiticity, spectrum, and purity exactly; a leaky (non-unitary)
    /// transport's effect shows up in the defect reads afterwards and is
    /// never repaired.
    ///
    /// `Transpose` path: the right frame moves by U and the left frame by
    /// the contragredient, Φ ← UΦ and Φ̃ ← U⁻ᵀΦ̃, so Γ ← U Γ U⁻¹ and the
    /// pairing Φ̃ᵀΦ is preserved (U⁻¹ from a full-pivot LU).
    /// @throws std::invalid_argument on a shape mismatch, or on the
    ///         `Transpose` path when U is singular (no contragredient).
    void applyTransport(const Eigen::MatrixXcd& transport);

    // ── mean-field self-consistency ─────────────────────────────────────

    /// The certificates-blind mean-field loop: for each of `steps`
    /// iterations, obtain h = `hamiltonian`(Γ) from the caller (classical
    /// geometry g is closed over by the caller), advance by `dt` via
    /// `evolve`, and record the per-iteration purity/Gaussianity
    /// certificate. Generalized Hartree-Fock dynamics: nonlinear in Γ but
    /// Gaussian-closed — the certificate measures that closure every step.
    /// The pure/mixed certificate path is chosen once, on entry: pure when
    /// purityDefect() ≤ `purityTolerance`. On the `Transpose` path the
    /// callback may return any complex generator (the self-consistent
    /// h(Γ) of a complex covariance is complex); both frames advance as in
    /// `evolve` and the certificate is max(purity, duality) per step.
    /// @throws std::invalid_argument when the callback returns a wrongly
    ///         shaped generator, or on the Hermitian-adjoint path a
    ///         non-Hermitian one (as `evolve`).
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

    /// The transpose-paired smeared Gram determinant
    /// ⟨a†(v_1)···a†(v_p) ã(w_p)···ã(w_1)⟩ = det(Wᵀ Γ V), with
    /// a†(v) = Σ v_i a_i† and the annihilator smeared by the transpose
    /// pairing, ã(w) = Σ w_i a_i (linear in w: the contraction by the dual
    /// mode, never conjugated). Columns of V create, columns of W
    /// annihilate. On a `Transpose` state it is the transition amplitude
    /// ⟨Ξ_L|···|Ξ_R⟩; with W = C̃ and V = C a left/right colour triad it is
    /// the complex determinant wedge det(C̃ᵀΓC). Mismatched column counts
    /// are exactly zero (number conservation).
    /// @throws std::invalid_argument when a frame's rows ≠ modeCount().
    [[nodiscard]] WickCertificateRead wickTransposeGramDeterminant(
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
    /// stored as informational channels and recomputed on load). The dual
    /// is recorded as `dual` ("hermitian-adjoint" or "transpose"); a
    /// `Transpose` state also stores both frames, and Γ is rebuilt from them
    /// on load. A record without `dual` is on the Hermitian-adjoint path.
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
    /// Γ ← ΦΦ̃ᵀ from the carried frames (`Transpose` path).
    void rebuildFromFrames();
    /// The premise residual of a read or certificate: the Hermiticity defect
    /// (Hermitian-adjoint path) or the duality defect (`Transpose` path).
    [[nodiscard]] double premiseDefect() const;
    /// The regime verified on Γ (shared by every certificate).
    [[nodiscard]] cobordism::CertificateRegime verifiedRegime() const;

    Eigen::MatrixXcd gamma_{};
    CovarianceDual dual_{CovarianceDual::HermitianAdjoint};
    // The frames of a `Transpose` state (0×0 otherwise).
    Eigen::MatrixXcd right_{};
    Eigen::MatrixXcd left_{};
    // Per-Γ-revision lazy defect caches (< 0 = not yet measured).
    mutable double hermiticityDefect_{-1.0};
    mutable double purityDefect_{-1.0};
    mutable double spectrumDefect_{-1.0};
    mutable double dualityDefect_{-1.0};
};

}  // namespace tessera::quantum
