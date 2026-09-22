// Lazy graded Fock oracle and boundary carrier.
//
// ─── Role ────────────────────────────────────────────────────────────────
//
// The one-particle edge space is h = span{|e⟩} (one two-level mode per
// edge, identified by modeId) and the global carrier is the fermionic Fock
// space F_-(h) = Λ•h. This engine represents vectors of finite stages of
// that carrier as a lazily evaluated expression DAG (directed acyclic
// graph), so generally entangled states are carried without eagerly
// allocating 2^M amplitudes and without a product-state ansatz: per-edge
// occupations are derived marginals of the global state, never stored
// per-edge state vectors. A stored product preparation is an optional
// boundary fixture and is labeled as such
// (`LazyFockEngine::boundaryProductFixture`).
//
// CovarianceState is the primary representation of the quasi-free sector:
// there, quadratic generators evolve Γ_ij = ⟨a_j†a_i⟩, never a Fock
// vector. This engine is that layer's dense reference and the carrier for
// explicitly non-Gaussian boundary data.
//
// ─── What lives here ─────────────────────────────────────────────────────
//
//   • LazyFockNode   — one immutable node of the expression DAG: vacuum,
//                      sparse occupation block, graded tensor product,
//                      local unitary/cobordism map, direct sum by conserved
//                      occupation/parity sector, antisymmetrized wedge.
//   • LazyFockState  — a value handle on a DAG root plus the state's
//                      accumulated discarded-norm bound and optional
//                      boundary-fixture label.
//   • LazyFockEngine — builders, lazy operator application, reads
//                      (amplitude / inner product / norm / covariance /
//                      free spectra / inductive compatibility), the
//                      exact-subexpression memo, and DAG serialization
//                      with content hashes.
//
// ─── Exact identities implemented (domains stated) ──────────────────────
//
//   • Graded tensor amplitude rule (any disjoint mode sets, arbitrary
//     interleaving in the global compilation order):
//       ⟨b_A ∪ b_B | ψ_A ⊗̂ ψ_B⟩ = ε(b_A,b_B) ⟨b_A|ψ_A⟩⟨b_B|ψ_B⟩,
//       ε(b_A,b_B) = (−1)^{#{(i,j) ∈ occ(b_A)×occ(b_B) : i > j}}.
//     This is the Koszul sign of sorting the concatenated wedge words and
//     generalizes the sign-free A-before-B identification of
//     FockDirectSum; it is strictly associative, so different
//     parenthesizations agree exactly after graded associators.
//   • Operator locality: a polynomial in {a_i, a_i† : i ∈ S} acts on
//     ⟨b| as a sum over its support columns with the two Koszul signs
//     ε(b_S, b_R) ε(c_S, b_R) — exact for even and odd operators, because
//     the ε convention carries the entire Jordan-Wigner bookkeeping.
//     Branchwise application below a graded tensor node is exact with no
//     sibling twist when the support lies in the left factor, and with
//     the Koszul parity twist (−1)^{|O||ψ_left|} (a scalar on
//     parity-definite left siblings) when an odd operator acts on the
//     right factor.
//   • Slater/wedge amplitudes: ⟨b| v_1∧…∧v_n⟩ = det[v_j(i)]_{i∈occ(b)}
//     (the Slater determinant), ‖v_1∧…∧v_n‖² = det[⟨v_i,v_j⟩], and the
//     normalized Slater covariance Γ = V(V†V)⁻¹V† — so a spectral
//     projector P initializes a quasi-free reference with Γ_ef = P_ef
//     exactly (`slaterFromProjector`, StructureExact given P² = P = P†).
//   • dΓ(L) = Σ_ij L_ij a_i†a_j applied at the bit level through
//     OccupationBitset::applyAnnihilation/applyCreation — the direct-sum
//     identity dΓ(L_A⊕L_B) = dΓ(L_A)⊗̂1 + 1⊗̂dΓ(L_B) and the
//     coupling-block hopping terms hold by construction and are
//     cross-checked against FockDirectSum::dGammaBlock; free occupation
//     subset-sum spectra delegate to cobordism::OccupationSpectra, never
//     re-derived.
//   • Vacuum embedding ι_M ψ = ψ ⊗̂ |0⟩: occupation keys are global
//     bitsets, so ι is the identity on every preexisting amplitude
//     (ε with an empty right word is +1) — amplitude preservation is by
//     construction and still verified through the API. The inductive
//     compatibility read reports ε_ι = ‖ι_M U_M − U_{M+1} ι_M‖ on the
//     active carried subspace as the top singular
//     value of the column-stacked defect.
//
// ─── Laziness contract ───────────────────────────────────────────────────
//
// A graded tensor node is expanded only when an applied operation's
// support crosses its partition; an operation supported inside one branch
// rewrites that branch and shares the untouched sibling node (verifiable
// through node ids / content hashes; `expansionCount` counts crossings).
// Exact subexpression expansions are memoized by content hash. Sector
// direct sums route reads by the conserved occupation/parity functional,
// and nodes carry definite occupation/parity where derivable so reads
// short-circuit out-of-sector amplitudes to exact zero (block sparsity).
//
// ─── Exactness and truncation ────────────────────────────────────────────
//
// Certification mode (the default) allows algebraically lossless rewrites
// only: expansion, merging duplicate keys, dropping exact zeros. Every
// scalar read carries the state's accumulated discarded norm — exactly
// 0.0 in certification mode — and a cobordism::Certificate. The optional
// truncation mode drops amplitudes at or below a stated threshold during
// materializations and accumulates an upper bound D on
// ‖ψ_exact − ψ̃‖₂ (triangle inequality across drops, operator-norm
// bounds across maps, norm-weighted sums across tensor products), so every
// reported amplitude satisfies |value − exact| ≤ D. Singular-value
// truncation is never silent, and allocation is bounded: enumerations
// refuse beyond `maxExpansionTerms` and dense exports beyond
// 2^kMaxDenseModes.
//
// ─── Mode order and hashes ───────────────────────────────────────────────
//
// The mode order is a compilation artifact: index i is position i of
// EdgeModeRegistry::canonicalModeOrder (see `fromRegistry`); relabeling
// applies OccupationBitset::permutationParity through `permuteModes`, and
// physical amplitudes are invariant. Content hashes chain
// mesh::Fingerprint::mix64 over the node's canonical byte content —
// `Fingerprint::fingerprintOf` itself is an order-independent XOR set
// hash, the wrong shape for order-sensitive expression hashing, so only
// the mixing primitive is reused.

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cobordism/Certificate.h"
#include "quantum/GradedFock.h"

namespace tessera::quantum {

/// The node vocabulary of the lazy expression DAG.
enum class LazyNodeKind {
    /// The vacuum Ω on the node's mode set (all modes empty).
    Vacuum,
    /// A sparse occupation block: finitely many global occupation bitsets
    /// with complex amplitudes.
    Occupation,
    /// The graded tensor product of two children over disjoint mode sets.
    GradedTensor,
    /// A local unitary/cobordism map held unevaluated over its child.
    LocalMap,
    /// A direct sum of children lying in distinct conserved sectors.
    SectorSum,
    /// An antisymmetrized wedge (Slater state) of one-particle orbitals.
    Wedge,
};

/// The conserved functional labelling a SectorSum's children.
enum class LazySectorKind {
    /// Total occupation number N (children have distinct definite N).
    Occupation,
    /// Fermion parity (−1)^N (children have distinct definite parity).
    Parity,
};

/// # LazyFockNode
///
/// One immutable node of the lazy Fock expression DAG. Nodes are built
/// only by LazyFockEngine, shared structurally (`shared_ptr`), and never
/// mutated after construction — an applied operation produces a new node
/// tree that shares every untouched branch of the old one.
///
/// Occupation keys are OccupationBitset over the engine's full mode count
/// — chunked, so arbitrary mode counts are supported — and
/// `modes()` is the sorted global mode subset the node may occupy (its
/// support restriction, used for the laziness and disjointness rules).
class LazyFockNode {
  public:
    using Complex = std::complex<double>;
    /// One sparse term: a global occupation bitset and its amplitude.
    using Term = std::pair<OccupationBitset, Complex>;
    using SparseRowOp = Eigen::SparseMatrix<Complex, Eigen::RowMajor>;
    using SparseColOp = Eigen::SparseMatrix<Complex, Eigen::ColMajor>;

    /// The node's kind.
    [[nodiscard]] LazyNodeKind kind() const noexcept { return kind_; }

    /// Sorted global mode indices this node covers.
    [[nodiscard]] const std::vector<std::size_t>& modes() const noexcept {
        return modes_;
    }

    /// Engine-wide mode universe size the node's keys are compiled over.
    [[nodiscard]] std::size_t universeModeCount() const noexcept {
        return universeModeCount_;
    }

    /// Process-unique node identity (for sharing/negative-control tests:
    /// an untouched sibling keeps its nodeId across an applied operation).
    [[nodiscard]] std::uint64_t nodeId() const noexcept { return nodeId_; }

    /// Deterministic content hash of the subexpression (stable across
    /// processes; the memoization and serialization key).
    [[nodiscard]] std::uint64_t contentHash() const noexcept {
        return contentHash_;
    }

    /// Children in order (GradedTensor: exactly {left, right}; SectorSum:
    /// sector-sorted children; LocalMap: exactly {child}; leaves: empty).
    [[nodiscard]] const std::vector<std::shared_ptr<const LazyFockNode>>&
    children() const noexcept {
        return children_;
    }

    /// Occupation terms, sorted by bitset (Occupation nodes; empty
    /// otherwise).
    [[nodiscard]] const std::vector<Term>& terms() const noexcept {
        return terms_;
    }

    /// Wedge orbitals as a |modes()| × n matrix in the node's local mode
    /// coordinates (row i = coefficient on modes()[i]).
    [[nodiscard]] const Eigen::MatrixXcd& orbitals() const noexcept {
        return orbitals_;
    }

    /// LocalMap support: sorted global mode indices the operator touches
    /// (a subset of modes()).
    [[nodiscard]] const std::vector<std::size_t>& supportModes()
        const noexcept {
        return supportModes_;
    }

    /// LocalMap operator over the 2^{|support|} support Fock basis
    /// (support-local bit k = k-th support mode in ascending global
    /// order), row-major for lazy per-amplitude rows.
    [[nodiscard]] const SparseRowOp& supportMatrixRows() const noexcept {
        return supportRows_;
    }

    /// The same LocalMap operator column-major, for state-side
    /// application.
    [[nodiscard]] const SparseColOp& supportMatrixCols() const noexcept {
        return supportCols_;
    }

    /// An upper bound on the LocalMap operator's spectral norm (exact
    /// ‖O‖₂ via SVD when the support dimension is small, else the
    /// Frobenius bound) — the factor by which an accumulated discarded
    /// norm grows through the map.
    [[nodiscard]] double operatorNormBound() const noexcept {
        return operatorNormBound_;
    }

    /// SectorSum functional kind.
    [[nodiscard]] LazySectorKind sectorKind() const noexcept {
        return sectorKind_;
    }

    /// SectorSum labels, parallel to children(): the child's definite
    /// occupation number, or its definite parity (+1/−1).
    [[nodiscard]] const std::vector<long long>& sectorLabels()
        const noexcept {
        return sectorLabels_;
    }

    /// Definite total occupation number of every term of this
    /// subexpression, or −1 when indefinite. Powers block-sparse reads:
    /// an amplitude at the wrong occupation is exactly zero without
    /// descending.
    [[nodiscard]] long long definiteOccupation() const noexcept {
        return definiteOccupation_;
    }

    /// Definite fermion parity (+1/−1), or 0 when indefinite.
    [[nodiscard]] int definiteParity() const noexcept {
        return definiteParity_;
    }

  private:
    friend class LazyFockEngine;

    LazyFockNode() = default;

    LazyNodeKind kind_{LazyNodeKind::Vacuum};
    std::vector<std::size_t> modes_{};
    std::size_t universeModeCount_{0};
    std::uint64_t nodeId_{0};
    std::uint64_t contentHash_{0};
    std::vector<std::shared_ptr<const LazyFockNode>> children_{};
    std::vector<Term> terms_{};
    Eigen::MatrixXcd orbitals_{};
    std::vector<std::size_t> supportModes_{};
    SparseRowOp supportRows_{};
    SparseColOp supportCols_{};
    double operatorNormBound_{0.0};
    LazySectorKind sectorKind_{LazySectorKind::Occupation};
    std::vector<long long> sectorLabels_{};
    long long definiteOccupation_{-1};
    int definiteParity_{0};
};

/// # LazyFockState
///
/// A value handle on one expression-DAG root: the represented vector of
/// F_-(h), the state's accumulated discarded-norm bound D (exactly 0.0 in
/// certification mode; in truncation mode an upper bound on
/// ‖ψ_exact − ψ̃‖₂ that every scalar read reports), and the optional
/// boundary-fixture label, set only by
/// LazyFockEngine::boundaryProductFixture.
class LazyFockState {
  public:
    LazyFockState() = default;

    /// Whether the handle points at a DAG (a default-constructed handle
    /// does not and is rejected by every engine method).
    [[nodiscard]] bool valid() const noexcept { return root_ != nullptr; }

    /// Root node identity (sharing tests).
    [[nodiscard]] std::uint64_t rootNodeId() const;
    /// Root content hash (memo/serialization key; replay-stable).
    [[nodiscard]] std::uint64_t contentHash() const;
    /// Root node kind.
    [[nodiscard]] LazyNodeKind kind() const;
    /// Sorted global mode indices the state covers.
    [[nodiscard]] std::vector<std::size_t> modes() const;
    /// nodeIds of the root's children, in order.
    [[nodiscard]] std::vector<std::uint64_t> childNodeIds() const;
    /// Content hashes of the root's children, in order.
    [[nodiscard]] std::vector<std::uint64_t> childContentHashes() const;
    /// Number of distinct nodes in the DAG (sharing-aware, so a shared
    /// subexpression counts once — the serialization also stores it once).
    [[nodiscard]] std::size_t nodeCount() const;
    /// Accumulated discarded-norm bound D (0.0 in certification mode).
    [[nodiscard]] double discardedNorm() const noexcept {
        return discardedNorm_;
    }
    /// Definite total occupation of the state, or −1 (block sparsity).
    [[nodiscard]] long long definiteOccupation() const;
    /// Definite fermion parity (+1/−1), or 0.
    [[nodiscard]] int definiteParity() const;
    /// The boundary-fixture label ("" for every non-fixture state).
    [[nodiscard]] const std::string& boundaryFixtureLabel() const noexcept {
        return boundaryFixtureLabel_;
    }
    /// Whether this state is a labeled product boundary fixture.
    [[nodiscard]] bool isBoundaryFixture() const noexcept {
        return !boundaryFixtureLabel_.empty();
    }

  private:
    friend class LazyFockEngine;

    LazyFockState(std::shared_ptr<const LazyFockNode> root,
                  double discardedNorm, std::string boundaryFixtureLabel)
        : root_(std::move(root)), discardedNorm_(discardedNorm),
          boundaryFixtureLabel_(std::move(boundaryFixtureLabel)) {}

    std::shared_ptr<const LazyFockNode> root_{};
    double discardedNorm_{0.0};
    std::string boundaryFixtureLabel_{};
};

/// A scalar read (amplitude, inner product, squared norm) together with
/// the accumulated discarded-norm bound of the state(s) it was read from
/// (exactly 0.0 in certification mode) and the Certificate grading it:
/// AlgebraicallyExact in certification mode, CertifiedNumerical with
/// residual = the discarded-norm bound in truncation mode. The bound is
/// absolute (an ℓ² bound on the state error, hence on any single
/// amplitude), a deviation from the Certificate default of relative
/// residuals.
struct LazyScalarRead {
    /// The evaluated scalar.
    std::complex<double> value{0.0, 0.0};
    /// Accumulated discarded-norm bound of the inputs (0.0 = exact).
    double discardedNorm{0.0};
    /// The certification record for this read.
    cobordism::Certificate certificate{};
};

/// The quasi-free/Slater reference initialized from a spectral projector
/// (Γ_ef = ⟨a_f†a_e⟩ = P_ef). StructureExact: exact given the verified
/// premise P² = P = P†, whose measured residual is on the certificate.
struct LazySlaterReference {
    /// The Slater determinant state (a Wedge node of the projector's
    /// rank-many occupied orbitals).
    LazyFockState state{};
    /// Number of occupied orbitals r = rank(P) = round(tr P).
    std::size_t rank{0};
    /// Measured premise residual max(‖P²−P‖, ‖P−P†‖)/max(1, ‖P‖)
    /// (Frobenius norms).
    double projectorResidual{0.0};
    /// StructureExact certificate carrying the premise residual.
    cobordism::Certificate certificate{};
};

/// A covariance read Γ_ef = ⟨a_f†a_e⟩/⟨ψ|ψ⟩ over the full engine mode
/// universe (modes outside the state's support are exactly vacuum rows /
/// columns). The certificate's residual is the measured Hermiticity
/// defect ‖Γ−Γ†‖/max(1,‖Γ‖) (closed-form Slater path: the trace defect
/// |tr Γ − n|/max(1,n)).
struct LazyCovarianceRead {
    /// The M×M covariance matrix Γ.
    Eigen::MatrixXcd matrix{};
    /// Accumulated discarded-norm bound of the state read from.
    double discardedNorm{0.0};
    /// The certification record.
    cobordism::Certificate certificate{};
};

/// The inductive compatibility read for the vacuum embedding ι_M:
/// ε_ι = ‖ι_M U_M − U_{M+1} ι_M‖ restricted to the active
/// carried subspace (the span of the supplied orthonormal occupation
/// basis states), computed as the top singular value of the
/// column-stacked defect.
struct LazyCompatibilityRead {
    /// ε_ι on the active carried subspace.
    double epsilon{0.0};
    /// Dimension of the active carried subspace actually spanned.
    std::size_t activeDimension{0};
    /// CertifiedNumerical record with the measured SVD residual.
    cobordism::Certificate certificate{};
};

/// One stage \f$ M \f$ of a refinement sequence
/// (`LazyFockEngine::inductiveLimit`): the modes the stage carries and the
/// map \f$ V_M \f$ on them.
///
/// The stages of a refinement sequence are nested — each stage's modes are a
/// subset of the next stage's — because the step from one to the next is the
/// vacuum embedding \f$ \iota_M(\psi) = \psi \mathbin{\hat\otimes} |0\rangle \f$,
/// which adds modes and changes no amplitude.
struct FockRefinementStage {
    /// The modes of \f$ \mathcal H_M \f$, ascending.
    std::vector<std::size_t> modes;
    /// The support of \f$ V_M \f$, a subset of `modes`.
    std::vector<std::size_t> support;
    /// \f$ V_M \f$, dense over the \f$ 2^{|support|} \f$ support Fock basis
    /// (the convention of `applyLocalMapDense`).
    Eigen::MatrixXcd map;
};

/// The inductive limit read over a refinement sequence
/// (`LazyFockEngine::inductiveLimit`).
///
/// The infinite Fock space is the direct limit
/// \f$ \mathcal F = \varinjlim(\mathcal H_M, \iota_M) \f$, and consistency of
/// the maps carried along it requires
/// \f$ \|\iota_M V_M - V_{M+1}\iota_M\| \to 0 \f$ over a refinement sequence.
/// That is a numerical certificate on a measured sequence, not a property any
/// single pair of stages can show: this read measures one defect per adjacent
/// pair, on one and the same active carried subspace, and reports the whole
/// sequence together with whether it falls.
struct LazyInductiveLimitRead {
    /// \f$ \varepsilon_\iota \f$ per adjacent pair of stages, in stage order:
    /// one fewer entry than there are stages.
    std::vector<double> defects;
    /// The full read of each adjacent pair, in the same order.
    std::vector<LazyCompatibilityRead> steps;
    /// The last defect of the sequence: how far the maps stand from
    /// compatible at the finest stage measured.
    double lastDefect{0.0};
    /// \f$ \max_k \varepsilon_{k+1}/\varepsilon_k \f$ over the sequence, with
    /// a vanishing predecessor giving \f$ +\infty \f$ unless its successor
    /// vanishes too: the worst step-to-step ratio, below one exactly when
    /// every step falls.
    double largestRatio{0.0};
    /// Every defect is strictly below its predecessor: the sequence falls.
    /// A sequence of one pair has nothing to fall against and reads false.
    bool falls{false};
    /// Dimension of the active carried subspace, the same at every step.
    std::size_t activeDimension{0};
    /// CertifiedNumerical record carrying the worst measured SVD residual of
    /// the steps.
    cobordism::Certificate certificate{};
};

/// # LazyFockEngine
///
/// The lazy graded Fock engine: builders, lazy operator application with
/// the crossing-only expansion rule, scalar/covariance/spectrum reads, the
/// exact-subexpression memo, the exact-certification / stated-truncation
/// switch, and DAG serialization with content hashes.
///
/// One engine instance fixes the mode universe (the compilation order:
/// index i = position i of EdgeModeRegistry::canonicalModeOrder — see
/// `fromRegistry`) and owns the memoization cache and counters. Not
/// synchronized — same threading contract as the analytic caches
/// (thread-private engines; a shared engine must be driven serially).
///
/// Density-operator boundary sectors are carried vectorized on a doubled
/// mode register (ket modes ⊕ bra modes) through the same six node kinds:
/// |ρ⟩⟩ = Σ ρ_ij |i⟩_ket ⊗̂ |j⟩_bra, with traces and occupation reads as
/// inner products against the vectorized identity. There is no dedicated
/// density node, and the arbitrary-mode-count carrier makes the doubled
/// register free.
///
/// This engine is not the production representation of the quasi-free
/// path: CovarianceState is, and this is its dense reference and the
/// carrier for explicitly non-Gaussian boundary data.
class LazyFockEngine {
  public:
    using Complex = std::complex<double>;

    /// Largest mode-universe size for which `denseVector` materializes
    /// the full 2^M vector (the ExteriorAlgebra::kMaxMatrixModes cap).
    static constexpr std::size_t kMaxDenseModes =
        ExteriorAlgebra::kMaxMatrixModes;
    /// Default refusal threshold for sparse expansions (number of terms —
    /// the OccupationSpectra::kDefaultMaxTerms convention).
    static constexpr std::size_t kDefaultMaxExpansionTerms =
        std::size_t{1} << 22;
    /// Largest LocalMap support (2^{|S|} operator dimension).
    static constexpr std::size_t kMaxSupportModes = 20;
    /// Serialization schema identifier.
    static constexpr const char* kSerializationSchema =
        "tessera.lazyfock.dag";
    /// Serialization schema version (readers reject unknown versions).
    static constexpr int kSerializationVersion = 1;

    /// Engine over `modeCount` modes (dim h; the carrier is Λ•C^M with
    /// dim 2^M, never allocated eagerly).
    explicit LazyFockEngine(std::size_t modeCount);

    /// Engine whose mode universe is the registry's deterministic
    /// compilation order: engine mode i = canonicalModeOrder()[i]
    /// (relabeling maps through `permuteModes` with
    /// EdgeModeRegistry::orderPermutation).
    [[nodiscard]] static LazyFockEngine fromRegistry(
        const EdgeModeRegistry& registry);

    /// dim h = M.
    [[nodiscard]] std::size_t modeCount() const noexcept {
        return modeCount_;
    }

    /// dim Λ•C^m = 2^m for a stage of `stageModeCount` ≤ 63 modes — the
    /// carrier-dimension identity on enumerable fixtures.
    /// @throws std::invalid_argument beyond 63 (the count itself would
    ///         overflow; the engine carries such stages lazily instead).
    [[nodiscard]] static std::uint64_t stageDimension(
        std::size_t stageModeCount);

    // ── exactness / truncation configuration ───────────────────────────

    /// Enter truncation mode: materializations drop amplitudes with
    /// |a| ≤ `threshold` and accumulate the discarded-norm bound;
    /// `normTolerance` is the declared budget the CertifiedNumerical read
    /// certificates hold against.
    /// @throws std::invalid_argument for negative/non-finite inputs.
    void setTruncationThreshold(double threshold, double normTolerance);

    /// Return to exact certification mode (algebraically lossless
    /// rewrites only; the default).
    void clearTruncation() noexcept;

    /// Whether the engine is in exact certification mode.
    [[nodiscard]] bool exactMode() const noexcept {
        return truncationThreshold_ == 0.0;
    }
    /// The truncation threshold (0.0 in certification mode).
    [[nodiscard]] double truncationThreshold() const noexcept {
        return truncationThreshold_;
    }
    /// The declared discarded-norm budget for truncation-mode holds().
    [[nodiscard]] double truncationNormTolerance() const noexcept {
        return truncationNormTolerance_;
    }

    /// Set the sparse-expansion refusal threshold (see
    /// kDefaultMaxExpansionTerms). Enumerations beyond it throw
    /// std::length_error instead of allocating an unbounded result.
    void setMaxExpansionTerms(std::size_t maxTerms) noexcept {
        maxExpansionTerms_ = maxTerms;
    }
    [[nodiscard]] std::size_t maxExpansionTerms() const noexcept {
        return maxExpansionTerms_;
    }

    // ── state builders ──────────────────────────────────────────────────

    /// The vacuum Ω over the full mode universe.
    [[nodiscard]] LazyFockState vacuum() const;

    /// The vacuum over a mode subset (sorted, deduplicated internally).
    /// @throws std::invalid_argument on an out-of-range mode.
    [[nodiscard]] LazyFockState vacuumOn(
        const std::vector<std::size_t>& modes) const;

    /// A sparse occupation block: term t occupies exactly
    /// `occupations[t]` (global mode indices) with amplitude
    /// `amplitudes[t]`. Duplicate keys merge (lossless); exact zeros
    /// drop (lossless); the node's mode set is `modes`.
    /// @throws std::invalid_argument on out-of-range/duplicate modes, a
    ///         term outside `modes`, a length mismatch, or a non-finite
    ///         amplitude.
    [[nodiscard]] LazyFockState occupationState(
        const std::vector<std::size_t>& modes,
        const std::vector<std::vector<std::size_t>>& occupations,
        const std::vector<Complex>& amplitudes) const;

    /// The antisymmetrized wedge v_1∧…∧v_n (a Slater state; n = 1 is a
    /// general one-particle state a†(v)Ω — e.g. the W/color state).
    /// `orbitals` is |modes| × n in the local coordinates of the sorted
    /// `modes` list. Duplicate/dependent orbitals give the exact zero
    /// vector (Pauli), reported by its zero norm, never an error.
    /// @throws std::invalid_argument on shape/mode errors or non-finite
    ///         entries.
    [[nodiscard]] LazyFockState wedgeState(
        const std::vector<std::size_t>& modes,
        const Eigen::MatrixXcd& orbitals) const;

    /// The optional quasi-free/Slater reference from a spectral
    /// projector: verifies P² = P = P† within `tolerance`
    /// (StructureExact premise), takes the rank-many occupied
    /// eigenvectors, and returns the Wedge state whose covariance is
    /// exactly Γ_ef = P_ef. Explicitly non-Gaussian sectors remain
    /// representable beside it; this reference never forces the state
    /// quasi-free.
    /// @throws std::invalid_argument when P is not M'×M' over `modes`
    ///         or the premise residual exceeds `tolerance`.
    [[nodiscard]] LazySlaterReference slaterFromProjector(
        const std::vector<std::size_t>& modes,
        const Eigen::MatrixXcd& projector, double tolerance) const;

    /// The graded tensor product a ⊗̂ b over disjoint mode sets (any
    /// interleaving in the global order — the ε sign rule in the header
    /// comment). Strictly associative: different parenthesizations of a
    /// multi-factor product agree amplitude-for-amplitude.
    /// @throws std::invalid_argument when the mode sets intersect.
    [[nodiscard]] LazyFockState gradedTensor(const LazyFockState& a,
                                             const LazyFockState& b) const;

    /// A direct sum of states lying in distinct definite sectors of the
    /// conserved functional `kind` (total occupation or parity), over a
    /// common mode set. Reads route by sector (block sparsity).
    /// @throws std::invalid_argument when a child's sector is indefinite,
    ///         two children share a sector, or mode sets differ.
    [[nodiscard]] LazyFockState sectorSum(
        const std::vector<LazyFockState>& children,
        LazySectorKind kind) const;

    /// The optional labeled product boundary fixture
    /// ∏_i (α_i + β_i a_i†) Ω — the only way to store a product
    /// preparation. `label` must be non-empty and travels on the state and
    /// through serialization.
    /// @throws std::invalid_argument on an empty label or shape errors.
    [[nodiscard]] LazyFockState boundaryProductFixture(
        const std::vector<std::size_t>& modes,
        const std::vector<Complex>& emptyAmplitudes,
        const std::vector<Complex>& occupiedAmplitudes,
        const std::string& label) const;

    /// The vacuum embedding ι: ψ ↦ ψ ⊗̂ |0⟩_{newModes} — the inductive-
    /// limit stage map. Preserves every preexisting amplitude exactly
    /// (keys are global; ε against an empty word is +1).
    /// @throws std::invalid_argument when `newModes` meets the state's
    ///         modes.
    [[nodiscard]] LazyFockState embedInVacuum(
        const LazyFockState& state,
        const std::vector<std::size_t>& newModes) const;

    // ── lazy operator application ───────────────────────────────────────

    /// Apply a local unitary/cobordism map given dense over the
    /// 2^{|support|} support Fock basis (support-local bit k = k-th
    /// support mode in ascending global order). Exact for even and odd
    /// operators; descends graded tensor branches and expands only on a
    /// partition crossing.
    /// @throws std::invalid_argument on support/shape errors or support
    ///         beyond kMaxSupportModes.
    [[nodiscard]] LazyFockState applyLocalMapDense(
        const LazyFockState& state,
        const std::vector<std::size_t>& supportModes,
        const Eigen::MatrixXcd& op) const;

    /// The same map given as COO triplets over the support Fock basis
    /// (the repository's sparse-crossing convention).
    /// @throws std::invalid_argument on an out-of-range triplet.
    [[nodiscard]] LazyFockState applyLocalMapCOO(
        const LazyFockState& state,
        const std::vector<std::size_t>& supportModes,
        const std::vector<std::int64_t>& rows,
        const std::vector<std::int64_t>& cols,
        const std::vector<Complex>& values) const;

    /// Apply a_mode† (an odd local map on one mode — the CAR sign
    /// bookkeeping is the OccupationBitset prefix-popcount rule).
    [[nodiscard]] LazyFockState applyCreation(const LazyFockState& state,
                                              std::size_t mode) const;

    /// Apply a_mode.
    [[nodiscard]] LazyFockState applyAnnihilation(const LazyFockState& state,
                                                  std::size_t mode) const;

    /// Apply dΓ(L) = Σ_ij L_ij a_i†a_j for an |S|×|S| one-particle block
    /// over `supportModes` — evaluated at the bit level through
    /// OccupationBitset creation/annihilation, so the support may be
    /// arbitrarily large (no 2^{|S|} operator is formed). Direct sums
    /// become graded tensor products and coupling blocks become hopping
    /// terms by construction (cross-checked against
    /// FockDirectSum::dGammaBlock in the suite).
    [[nodiscard]] LazyFockState applyDGamma(
        const LazyFockState& state,
        const std::vector<std::size_t>& supportModes,
        const Eigen::MatrixXcd& oneParticle) const;

    /// Materialize the state as a single sparse Occupation node —
    /// algebraically lossless in certification mode; in truncation mode
    /// drops |a| ≤ threshold and adds the dropped chunk's ℓ² norm to the
    /// state's discarded-norm bound.
    [[nodiscard]] LazyFockState materialize(const LazyFockState& state) const;

    /// The exact signed mode relabeling: bit perm[i] of every key of the
    /// result corresponds to bit i, with
    /// OccupationBitset::permutationParity applied per term — physical
    /// amplitudes are invariant under relabeling once this parity is
    /// applied (EdgeModeRegistry::orderPermutation is the intended source
    /// of `perm`).
    [[nodiscard]] LazyFockState permuteModes(
        const LazyFockState& state,
        const std::vector<std::size_t>& perm) const;

    // ── reads ───────────────────────────────────────────────────────────

    /// ⟨b|ψ⟩ for the global basis state occupying exactly
    /// `occupiedModes`. Occupied modes outside the state's mode set give
    /// exact zero (the state is vacuum there). Lazy: descends the DAG
    /// without materializing (Slater amplitudes are n×n determinants;
    /// out-of-sector reads short-circuit).
    [[nodiscard]] LazyScalarRead amplitude(
        const LazyFockState& state,
        const std::vector<std::size_t>& occupiedModes) const;

    /// ⟨a|b⟩ (antilinear in `a`). Wedge/Wedge pairs use the exact Gram
    /// identity det[⟨v_i, w_j⟩]; an Occupation side drives the sum
    /// without expanding the other side; general pairs expand through
    /// the memo. The reported discardedNorm is the two states' combined
    /// bound scaled by the partners' norms.
    [[nodiscard]] LazyScalarRead innerProduct(const LazyFockState& a,
                                              const LazyFockState& b) const;

    /// ‖ψ‖² (Wedge: det Gram exactly; GradedTensor: product;
    /// SectorSum: sector-orthogonal sum).
    [[nodiscard]] LazyScalarRead normSquared(const LazyFockState& state) const;

    /// Γ_ef = ⟨a_f†a_e⟩/‖ψ‖² over the full mode universe. Wedge states
    /// use the exact closed form Γ = V(V†V)⁻¹V† (so
    /// `slaterFromProjector(P).state` reads back Γ = P exactly);
    /// general states read through the sparse expansion.
    [[nodiscard]] LazyCovarianceRead covarianceMatrix(
        const LazyFockState& state) const;

    /// The full dense Fock vector over the mode universe, index
    /// n(b) = Σ_i b_i 2^i (the ExteriorAlgebra basis convention) — the
    /// crossover comparison export.
    /// @throws std::invalid_argument beyond kMaxDenseModes (a literal
    ///         2^M allocation is out of scope).
    [[nodiscard]] Eigen::VectorXcd denseVector(
        const LazyFockState& state) const;

    /// The free N-particle spectrum of dΓ(L): eigenvalues of the
    /// one-particle block, then occupation subset sums delegated to
    /// cobordism::OccupationSpectra. CertifiedNumerical: residual = the
    /// measured max eigen-residual ‖Lv−λv‖/max(1,‖L‖).
    [[nodiscard]] cobordism::CertifiedVector freeSpectrum(
        const Eigen::MatrixXcd& oneParticle, int particles) const;

    /// Subset sums of an already-known one-particle spectrum (pure
    /// OccupationSpectra delegation). AlgebraicallyExact: the residual is
    /// the measured elementwise gap against the independent
    /// OccupationSpectra::directSumSubsetSums split evaluation.
    [[nodiscard]] cobordism::CertifiedVector freeSpectrumFromEigenvalues(
        const std::vector<Complex>& oneParticleSpectrum,
        int particles) const;

    /// The inductive compatibility read ε_ι = ‖ι_M U_M − U_{M+1} ι_M‖ on the
    /// active carried subspace: U_M is a local map on the stage-M
    /// carrier (support ⊆ `stageModes`), U_{M+1} on the extended carrier
    /// (support ⊆ `extendedModes` ⊇ `stageModes`), and the active
    /// subspace is the span of the orthonormal basis states occupying
    /// `activeBasis[t]` ⊆ stageModes. ε_ι is the top singular value of
    /// the column-stacked defect.
    /// @throws std::invalid_argument on containment violations.
    [[nodiscard]] LazyCompatibilityRead inductiveCompatibility(
        const std::vector<std::size_t>& stageModes,
        const std::vector<std::size_t>& extendedModes,
        const std::vector<std::size_t>& stageSupport,
        const Eigen::MatrixXcd& stageOp,
        const std::vector<std::size_t>& extendedSupport,
        const Eigen::MatrixXcd& extendedOp,
        const std::vector<std::vector<std::size_t>>& activeBasis) const;

    /// The inductive limit read over a whole refinement sequence: one
    /// `inductiveCompatibility` per adjacent pair of `stages`, all on the same
    /// active carried subspace, with the defect sequence and whether it falls
    /// (see `LazyInductiveLimitRead`).
    ///
    /// The whitepaper's consistency condition,
    /// \f$ \|\iota_M V_M - V_{M+1}\iota_M\| \to 0 \f$, is a statement about a
    /// sequence. One pair of stages can only give one number, which no more
    /// establishes a limit than one term establishes a series; this is the
    /// entry point that measures the sequence.
    ///
    /// `activeBasis` lists the occupation states spanning the carried subspace
    /// the comparison is made on. They must lie inside the FIRST stage's
    /// modes, so that one and the same subspace is carried through every
    /// embedding of the sequence and the defects are commensurable.
    /// @throws std::invalid_argument for fewer than two stages, for stages
    ///         that are not nested, and as `inductiveCompatibility`.
    [[nodiscard]] LazyInductiveLimitRead inductiveLimit(
        const std::vector<FockRefinementStage>& stages,
        const std::vector<std::vector<std::size_t>>& activeBasis) const;

    // ── memo / counters ─────────────────────────────────────────────────

    /// Times a graded tensor partition was expanded because an applied
    /// operation crossed it (the only expansion trigger).
    [[nodiscard]] std::uint64_t expansionCount() const noexcept {
        return expansionCount_;
    }
    /// Memoized-expansion hits since construction / clearMemo.
    [[nodiscard]] std::uint64_t memoHits() const noexcept { return memoHits_; }
    /// Memoized-expansion misses.
    [[nodiscard]] std::uint64_t memoMisses() const noexcept {
        return memoMisses_;
    }
    /// Live memo entries.
    [[nodiscard]] std::size_t memoSize() const noexcept {
        return memo_.size();
    }
    /// Drop every memo entry (cold-versus-memoized comparisons).
    void clearMemo() noexcept;

    // ── serialization ───────────────────────────────────────────────────

    /// Serialize the state's DAG as strict JSON without flattening it:
    /// nodes are listed once in topological order and referenced by
    /// index, so shared subexpressions stay shared; every node carries
    /// its content hash; amplitudes round-trip bit-exactly (17
    /// significant digits). Includes the discarded-norm bound and the
    /// boundary-fixture label.
    [[nodiscard]] std::string serialize(const LazyFockState& state) const;

    /// Rebuild a state from `serialize` output: verifies the schema
    /// name/version, the mode universe, and every node's recomputed
    /// content hash against the stored one (rejecting a tampered or
    /// drifted checkpoint), and reproduces expressions and amplitudes
    /// exactly.
    /// @throws std::invalid_argument on malformed input, a schema
    ///         mismatch, or a content-hash mismatch.
    [[nodiscard]] LazyFockState deserialize(const std::string& json) const;

  private:
    struct Expansion;      // sorted global sparse terms (defined in .cpp)
    struct OperatorSpec;   // one application request (defined in .cpp)

    using NodePtr = std::shared_ptr<const LazyFockNode>;

    // node factories (canonicalize, derive sectors, hash)
    [[nodiscard]] NodePtr makeVacuum(std::vector<std::size_t> modes) const;
    [[nodiscard]] NodePtr makeOccupation(
        std::vector<std::size_t> modes,
        std::vector<LazyFockNode::Term> terms) const;
    [[nodiscard]] NodePtr makeTensor(NodePtr left, NodePtr right) const;
    [[nodiscard]] NodePtr makeSectorSum(
        std::vector<NodePtr> children, LazySectorKind kind) const;
    [[nodiscard]] NodePtr makeLocalMap(
        NodePtr child, std::vector<std::size_t> supportModes,
        LazyFockNode::SparseColOp op) const;
    [[nodiscard]] NodePtr makeWedge(std::vector<std::size_t> modes,
                                    Eigen::MatrixXcd orbitals) const;

    // evaluation core
    [[nodiscard]] Complex amplitudeOf(const LazyFockNode& node,
                                      const OccupationBitset& key) const;
    [[nodiscard]] std::shared_ptr<const Expansion> expand(
        const NodePtr& node) const;
    [[nodiscard]] double normSquaredOf(const NodePtr& node) const;
    [[nodiscard]] NodePtr applyOperator(const NodePtr& node,
                                        const OperatorSpec& op,
                                        double* drops) const;
    [[nodiscard]] NodePtr applyToExpansion(const Expansion& expansion,
                                           const std::vector<std::size_t>& modes,
                                           const OperatorSpec& op,
                                           double* drops) const;

    // shared helpers
    void validateState(const LazyFockState& state) const;
    [[nodiscard]] std::vector<std::size_t> canonicalModes(
        const std::vector<std::size_t>& modes, const char* what) const;
    [[nodiscard]] cobordism::Certificate readCertificate(
        double discardedNorm) const;
    [[nodiscard]] LazyFockState wrap(NodePtr node, double discardedNorm,
                                     std::string label = {}) const;

    std::size_t modeCount_{0};
    double truncationThreshold_{0.0};
    double truncationNormTolerance_{0.0};
    std::size_t maxExpansionTerms_{kDefaultMaxExpansionTerms};
    mutable std::unordered_map<std::uint64_t, std::shared_ptr<const Expansion>>
        memo_{};
    mutable std::uint64_t expansionCount_{0};
    mutable std::uint64_t memoHits_{0};
    mutable std::uint64_t memoMisses_{0};
};

}  // namespace tessera::quantum
