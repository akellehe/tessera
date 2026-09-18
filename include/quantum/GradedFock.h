// Oriented exterior-algebra and graded-tensor primitives for the finite
// stages of a generally entangled fermionic Fock space.
//
// ─── What lives here ─────────────────────────────────────────────────────
//
//   • OccupationBitset    — an exterior basis state as a chunked occupation
//                           bitset with the exact prefix-popcount sign rule.
//   • ExteriorAlgebra     — creation/annihilation/number/parity matrices,
//                           wedge and contraction, occupation-sector
//                           projectors, dΓ, and the signed mode-permutation
//                           representation on Λ•C^M.
//   • GradedTensorComplex — the graded chain/tensor differential
//                           d(a⊗b) = da⊗b + (−1)^{deg a} a⊗db for actual
//                           product complexes, with per-degree Hodge
//                           Laplacians.
//   • FockDirectSum       — the Fock direct-sum functor
//                           F(h_A ⊕ h_B) ≅ F(h_A) ⊗ F(h_B): graded operator
//                           lifts, the graded swap, and dΓ of a block
//                           one-particle operator (hopping terms).
//   • EdgeModeRegistry    — the edge-mode basis/reorientation convention and
//                           the deterministic compilation order derived from
//                           oriented component lineage, including the parity
//                           map under vertex relabeling.
//
// ─── Exact identities implemented (and tested to double round-off) ──────
//
//   • The canonical anticommutation relations (CAR):
//     {a_i, a_j} = 0,  {a_i†, a_j†} = 0,  {a_i, a_j†} = δ_ij.
//   • dim Λ•C^M = 2^M.
//   • ‖v_1∧…∧v_n‖² = det[⟨v_i, v_j⟩]; duplicate complete one-particle
//     modes wedge to exactly zero.
//   • Graded exchange: odd/odd = −1; every other elementary parity
//     combination = +1.
//   • Graded Leibniz: d(a⊗b) = da⊗b + (−1)^{deg a} a⊗db, hence d∘d = 0 and
//     Δ_{A⊗B} = Δ_A⊗1 + 1⊗Δ_B blockwise (Künneth at the Hodge level).
//   • dΓ(L_A ⊕ L_B) = dΓ(L_A)⊗1 + 1⊗dΓ(L_B) under the direct-sum
//     identification; coupling blocks become hopping terms
//     Σ_{i∈A, j∈B} C_ij a_i†a_j + h.c.
//
// ─── Mode order is a compilation artifact ────────────────────────────────
//
// The abstract exterior algebra Λ•h is order-independent: no Kasteleyn
// orientation is required. A total order on the modes is chosen only to
// compile basis states into bitsets and operators into matrices, and the
// deterministic order comes from oriented component lineage
// (EdgeModeRegistry::canonicalModeOrder). A vertex relabeling rebuilds the
// canonical order and applies the corresponding permutation parity
// (OccupationBitset::permutationParity /
// ExteriorAlgebra::modePermutationMatrix), under which every physical
// amplitude is invariant.
//
// ─── Edge-mode semantics ─────────────────────────────────────────────────
//
// Each edge indexes one two-level mode factor span{|0⟩,|1⟩} inside the
// global exterior Fock space Λ•h — identified by a modeId, never by stored
// per-edge state vectors. An Edge's geometric datum remains exactly one
// complex length; nothing here adds quantum state to the edge record. A
// per-edge occupation (or Bloch vector) is a derived marginal of the global
// state, not a stored product state; a product preparation is an optional
// boundary fixture and must be labeled as such by its owner.
//
// Reorientation convention (documented once, used everywhere): reversing an
// edge fixes the mode factor pointwise on {|0⟩,|1⟩} (no basis permutation,
// no conjugation of the factor) and multiplies the one-particle embedding
// vector of the mode by −1 — equivalently a_e ↦ −a_e, a_e† ↦ −a_e†. This
// map preserves the CAR and every occupation observable (number, parity,
// sector projectors) and flips the sign of one-particle amplitudes; the
// stored orientationSign flips so that the effective oriented edge
// (orientationSign × stored direction) is unchanged under a pure storage
// reversal.

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }

namespace tessera::quantum {

/// # OccupationBitset
///
/// An exterior basis state of Λ•C^M as an occupation bitset over M modes,
/// chunked into 64-bit machine words so the representation is correct for
/// arbitrary mode counts: one word up to the machine-word threshold,
/// `⌈M/64⌉` words above it, with the same prefix-popcount sign rule in
/// both regimes.
///
/// The creation sign is exactly
/// \f$ (-1)^{\mathrm{popcount}(b\,\&\,((1\ll i)-1))} \f$ — the parity of the
/// number of occupied modes strictly below mode `i` in the compilation
/// order. The bitset ↔ wedge dictionary is
/// \f$ |b\rangle = e_{i_1}\wedge\cdots\wedge e_{i_k} \f$ with
/// \f$ i_1 < \cdots < i_k \f$ the occupied modes in ascending order.
///
/// The mode order underlying bit positions is a compilation artifact
/// (see EdgeModeRegistry); `permuted` / `permutationParity` implement the
/// exact signed action of a mode relabeling on basis states.
class OccupationBitset {
  public:
    /// Bits per storage chunk (one machine word).
    static constexpr std::size_t kBitsPerChunk = 64;

    /// The vacuum bitset (all modes empty) over `modeCount` modes.
    explicit OccupationBitset(std::size_t modeCount);

    /// Build from an explicit list of occupied modes.
    /// @throws std::invalid_argument on an out-of-range or duplicate mode.
    [[nodiscard]] static OccupationBitset fromOccupiedModes(
        std::size_t modeCount, const std::vector<std::size_t>& modes);

    /// Build from a Fock basis index (mode `i` occupied iff bit `i` of
    /// `index` is set). Requires `modeCount <= 64` and
    /// `index < 2^modeCount` (for `modeCount == 64`, any index).
    /// @throws std::invalid_argument otherwise.
    [[nodiscard]] static OccupationBitset fromIndex(std::size_t modeCount,
                                                    std::uint64_t index);

    /// Number of modes M (bit positions 0..M-1).
    [[nodiscard]] std::size_t modeCount() const noexcept { return modeCount_; }

    /// Number of 64-bit storage chunks, `⌈M/64⌉`.
    [[nodiscard]] std::size_t chunkCount() const noexcept {
        return chunks_.size();
    }

    /// Raw storage chunks, least-significant chunk first (chunk c holds
    /// modes 64c .. 64c+63). Exposed for tests and scaling checks.
    [[nodiscard]] const std::vector<std::uint64_t>& chunks() const noexcept {
        return chunks_;
    }

    /// Occupation of `mode` (true = occupied).
    /// @throws std::invalid_argument if `mode >= modeCount()`.
    [[nodiscard]] bool test(std::size_t mode) const;

    /// Occupy `mode` unconditionally (no sign bookkeeping — use
    /// `applyCreation` for the signed CAR action).
    void set(std::size_t mode);

    /// Empty `mode` unconditionally (no sign bookkeeping).
    void reset(std::size_t mode);

    /// Total occupation number N = popcount(b).
    [[nodiscard]] std::size_t count() const noexcept;

    /// Fermion parity \f$ (-1)^N \f$ as +1 / −1.
    [[nodiscard]] int parity() const noexcept;

    /// Number of occupied modes strictly below `mode`:
    /// `popcount(b & ((1 << mode) - 1))`, computed chunk-wise.
    /// @throws std::invalid_argument if `mode > modeCount()`
    /// (`mode == modeCount()` is allowed and returns `count()`).
    [[nodiscard]] std::size_t prefixPopcount(std::size_t mode) const;

    /// Apply \f$ a_i^\dagger \f$ at the bit level: returns 0 and leaves the
    /// state unchanged when `mode` is already occupied (Pauli exclusion);
    /// otherwise occupies it and returns the exact creation sign
    /// \f$ (-1)^{\mathrm{prefixPopcount}(i)} \in \{+1,-1\} \f$.
    int applyCreation(std::size_t mode);

    /// Apply \f$ a_i \f$ at the bit level: returns 0 and leaves the state
    /// unchanged when `mode` is empty; otherwise empties it and returns
    /// \f$ (-1)^{\mathrm{prefixPopcount}(i)} \f$.
    int applyAnnihilation(std::size_t mode);

    /// Fock basis index Σ_i b_i 2^i. Requires `modeCount() <= 64`.
    /// @throws std::invalid_argument otherwise.
    [[nodiscard]] std::uint64_t toIndex() const;

    /// Occupied modes in ascending order (the wedge word of the state).
    [[nodiscard]] std::vector<std::size_t> occupiedModes() const;

    /// The relabeled bitset: bit `perm[i]` of the result equals bit `i` of
    /// `*this`. `perm` must be a bijection on {0,…,M−1}.
    /// @throws std::invalid_argument if `perm` is not a bijection of size M.
    [[nodiscard]] OccupationBitset permuted(
        const std::vector<std::size_t>& perm) const;

    /// The exact parity picked up by this basis state under the mode
    /// relabeling `perm`: with occupied modes \f$ i_1<\cdots<i_k \f$,
    /// \f$ a_{i_1}^\dagger\cdots a_{i_k}^\dagger\Omega \mapsto
    ///     a_{\pi(i_1)}^\dagger\cdots a_{\pi(i_k)}^\dagger\Omega
    ///   = \varepsilon\,|\pi(b)\rangle \f$
    /// where \f$ \varepsilon = (-1)^{\#\mathrm{inversions}(\pi(i_1),\ldots,
    /// \pi(i_k))} \f$. Returns +1 / −1.
    /// @throws std::invalid_argument if `perm` is not a bijection of size M.
    [[nodiscard]] int permutationParity(
        const std::vector<std::size_t>& perm) const;

    [[nodiscard]] bool operator==(const OccupationBitset& other) const noexcept;

    /// "|b_{M-1} … b_1 b_0⟩" occupation string, mode 0 rightmost.
    [[nodiscard]] std::string str() const;

  private:
    static void validatePermutation(const std::vector<std::size_t>& perm,
                                    std::size_t modeCount);

    std::vector<std::uint64_t> chunks_{};
    std::size_t modeCount_{0};
};

/// # ExteriorAlgebra
///
/// The full exterior algebra \f$ \Lambda^\bullet \mathbb{C}^M \f$ over an
/// M-mode one-particle space, with its CAR operator layer as explicit
/// sparse matrices on the \f$ 2^M \f$-dimensional Fock basis
/// \f$ |b\rangle,\ b \in \{0,1\}^M \f$, indexed by
/// \f$ n(b) = \sum_i b_i 2^i \f$ (mode 0 = least-significant bit).
///
/// Exact identities carried by this class (all integer/algebraic, tested to
/// double round-off):
///   • \f$ \{a_i,a_j\} = \{a_i^\dagger,a_j^\dagger\} = 0 \f$,
///     \f$ \{a_i,a_j^\dagger\} = \delta_{ij} \f$;
///   • \f$ \dim \Lambda^\bullet \mathbb{C}^M = 2^M \f$;
///   • \f$ \|v_1\wedge\cdots\wedge v_n\|^2 = \det[\langle v_i,v_j\rangle] \f$
///     with \f$ \langle v,w\rangle = \sum_k \overline{v_k} w_k \f$;
///   • duplicate complete one-particle modes wedge to exactly zero.
///
/// The matrix layer is dense/sparse over up to `kMaxMatrixModes` modes —
/// sufficient for exact fixtures; arbitrary mode counts are supported at the
/// data-structure level by OccupationBitset. Lazy large-scale carriers live
/// in LazyFockEngine; nothing here allocates a full Fock state beyond the
/// requested operators.
///
/// The mode order is a compilation artifact of the abstract (order-free)
/// exterior algebra; `modePermutationMatrix` provides the exact signed
/// unitary intertwining two compilations. No Kasteleyn orientation is
/// required.
class ExteriorAlgebra {
  public:
    using Complex = std::complex<double>;
    using SparseOp = Eigen::SparseMatrix<Complex>;

    /// Largest mode count for which the 2^M-dimensional matrix layer may be
    /// materialized (2^24 basis states). OccupationBitset has no such limit.
    static constexpr std::size_t kMaxMatrixModes = 24;

    /// Algebra over `modeCount` modes.
    /// @throws std::invalid_argument if `modeCount > kMaxMatrixModes`.
    explicit ExteriorAlgebra(std::size_t modeCount);

    [[nodiscard]] std::size_t modeCount() const noexcept { return modeCount_; }

    /// \f$ \dim \Lambda^\bullet \mathbb{C}^M = 2^M \f$.
    [[nodiscard]] std::size_t fockDimension() const noexcept { return dim_; }

    /// Creation matrix \f$ a_i^\dagger \f$: entries
    /// \f$ \langle b\cup\{i\}|a_i^\dagger|b\rangle =
    ///     (-1)^{\mathrm{prefixPopcount}_b(i)} \f$ for \f$ i \notin b \f$.
    [[nodiscard]] SparseOp creationMatrix(std::size_t mode) const;

    /// Annihilation matrix \f$ a_i \f$ (adjoint of `creationMatrix`).
    [[nodiscard]] SparseOp annihilationMatrix(std::size_t mode) const;

    /// Mode-occupation matrix \f$ n_i = a_i^\dagger a_i \f$ (diagonal 0/1).
    [[nodiscard]] SparseOp numberMatrix(std::size_t mode) const;

    /// Total number operator \f$ N = \sum_i n_i \f$ (diagonal).
    [[nodiscard]] SparseOp totalNumberMatrix() const;

    /// Fermion parity \f$ (-1)^N \f$ (diagonal ±1).
    [[nodiscard]] SparseOp parityMatrix() const;

    /// Projector onto total occupation N = `occupation` (diagonal 0/1).
    /// Zero matrix when `occupation > modeCount()`.
    [[nodiscard]] SparseOp sectorProjector(std::size_t occupation) const;

    /// Occupation-sector projector for a mode subset: projects onto basis
    /// states whose occupation restricted to `modes` equals `occupation`.
    /// With a three-mode subset this yields the exact
    /// \f$ \Lambda^0,\Lambda^1,\Lambda^2,\Lambda^3 \f$ sector projectors of
    /// that three-mode factor (occupation-number projectors).
    /// @throws std::invalid_argument on out-of-range or duplicate modes.
    [[nodiscard]] SparseOp subsetSectorProjector(
        const std::vector<std::size_t>& modes, std::size_t occupation) const;

    /// The vacuum \f$ \Omega = |0\cdots0\rangle \f$ as a dense Fock vector.
    [[nodiscard]] Eigen::VectorXcd vacuumState() const;

    /// The basis vector \f$ |b\rangle \f$ for an occupation bitset (must
    /// have matching modeCount).
    [[nodiscard]] Eigen::VectorXcd basisState(const OccupationBitset& b) const;

    /// Smeared creation operator \f$ a^\dagger(v) = \sum_i v_i a_i^\dagger \f$
    /// (linear in `v`); `v` must have size M.
    [[nodiscard]] SparseOp creationOperator(const Eigen::VectorXcd& v) const;

    /// Smeared annihilation operator
    /// \f$ a(v) = \sum_i \overline{v_i}\, a_i \f$ (antilinear in `v`), so
    /// that \f$ \{a(v), a^\dagger(w)\} = \langle v, w\rangle \f$.
    [[nodiscard]] SparseOp annihilationOperator(const Eigen::VectorXcd& v) const;

    /// The wedge product as a Fock vector:
    /// \f$ v_1\wedge\cdots\wedge v_n = a^\dagger(v_1)\cdots a^\dagger(v_n)
    /// \Omega \f$ (rightmost factor applied first). Satisfies
    /// \f$ \|v_1\wedge\cdots\wedge v_n\|^2 = \det[\langle v_i,v_j\rangle] \f$
    /// exactly; repeating a complete one-particle mode gives exactly zero.
    [[nodiscard]] Eigen::VectorXcd wedge(
        const std::vector<Eigen::VectorXcd>& vectors) const;

    /// Interior product (contraction) \f$ \iota_w \psi = a(w)\,\psi \f$ —
    /// the odd antiderivation adjoint to wedging with `w`.
    [[nodiscard]] Eigen::VectorXcd contract(const Eigen::VectorXcd& w,
                                            const Eigen::VectorXcd& state) const;

    /// Second quantization \f$ d\Gamma(L) = \sum_{ij} L_{ij}\,
    /// a_i^\dagger a_j \f$ of an M×M one-particle block matrix — the
    /// number-preserving quadratic/hopping operator. For Hermitian L its
    /// spectrum is exactly the set of occupation subset sums of the
    /// one-particle spectrum.
    /// @throws std::invalid_argument if L is not M×M.
    [[nodiscard]] SparseOp dGamma(const Eigen::MatrixXcd& oneParticle) const;

    /// The exact signed permutation unitary \f$ U_\pi \f$ compiled from a
    /// mode relabeling: \f$ U_\pi|b\rangle = \varepsilon_\pi(b)\,
    /// |\pi(b)\rangle \f$ with \f$ \varepsilon_\pi(b) \f$ from
    /// OccupationBitset::permutationParity, so that
    /// \f$ U_\pi a_i^\dagger U_\pi^\dagger = a_{\pi(i)}^\dagger \f$. All
    /// physical amplitudes are invariant under relabeling once this parity
    /// is applied.
    [[nodiscard]] SparseOp modePermutationMatrix(
        const std::vector<std::size_t>& perm) const;

  private:
    void validateMode(std::size_t mode) const;

    std::size_t modeCount_{0};
    std::size_t dim_{1};
};

/// # GradedTensorComplex
///
/// The graded tensor product of two finite chain complexes
/// \f$ (A_\bullet, \partial^A) \f$ and \f$ (B_\bullet, \partial^B) \f$:
/// \f$ C_n = \bigoplus_{p+q=n} A_p \otimes B_q \f$ with the graded Leibniz
/// differential
/// \f$ d(a\otimes b) = da\otimes b + (-1)^{\deg a}\, a\otimes db \f$,
/// assembled blockwise as
/// \f$ \partial^A_p \otimes 1_{B_q} \f$ into block \f$ (p{-}1,q) \f$ and
/// \f$ (-1)^p\, 1_{A_p} \otimes \partial^B_q \f$ into block
/// \f$ (p,q{-}1) \f$.
///
/// This is the chain complex of an actual product cell complex (cubical /
/// CW products satisfy \f$ C(X\times Y) = C(X)\otimes C(Y) \f$ on the
/// nose), so product-complex Hodge fixtures must match this construction
/// exactly. Exact consequences carried by the sign rule:
///   • \f$ d\circ d = 0 \f$;
///   • the cross terms in \f$ \Delta = dd^\dagger + d^\dagger d \f$ cancel,
///     giving \f$ \Delta_n = \bigoplus_{p+q=n}(\Delta^A_p\otimes 1 +
///     1\otimes\Delta^B_q) \f$ — the degree-n Hodge spectrum is the multiset
///     of pairwise sums (Künneth at the Hodge level, identity metrics).
///
/// Conventions (documented compilation choices):
///   • `diff[k]` is \f$ \partial_{k+1}: C_{k+1}\to C_k \f$ with shape
///     `dims[k] × dims[k+1]`;
///   • blocks of \f$ C_n \f$ are ordered by ascending p;
///   • within block \f$ (p,q) \f$ the index is `i_a * dimB_q + i_b`
///     (Kronecker convention `kron(A-side, B-side)`).
class GradedTensorComplex {
  public:
    /// Build from the two factor complexes.
    /// `dimsX[k]` is \f$ \dim X_k \f$; `diffX[k]` is
    /// \f$ \partial_{k+1}: X_{k+1}\to X_k \f$ (so `diffX.size() ==
    /// dimsX.size() - 1`; both may describe a single-degree complex with no
    /// differentials). Each factor's \f$ \partial\circ\partial = 0 \f$ is
    /// validated with max-entry tolerance `boundaryTolerance` (default 0 —
    /// exact, appropriate for integer boundary matrices).
    /// @throws std::invalid_argument on shape mismatch or a violated
    ///         \f$ \partial\circ\partial = 0 \f$.
    GradedTensorComplex(std::vector<std::size_t> dimsA,
                        std::vector<Eigen::MatrixXcd> diffA,
                        std::vector<std::size_t> dimsB,
                        std::vector<Eigen::MatrixXcd> diffB,
                        double boundaryTolerance = 0.0);

    /// Top degree \f$ p_{\max} + q_{\max} \f$ of the product complex.
    [[nodiscard]] std::size_t maxDegree() const noexcept;

    /// \f$ \dim C_n = \sum_{p+q=n} \dim A_p \cdot \dim B_q \f$
    /// (0 when `degree > maxDegree()`).
    [[nodiscard]] std::size_t chainDimension(std::size_t degree) const;

    /// The (p, q) block labels of \f$ C_n \f$ in storage order
    /// (ascending p, only in-range factors).
    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> blocks(
        std::size_t degree) const;

    /// The graded Leibniz differential
    /// \f$ \partial_n: C_n \to C_{n-1} \f$, `1 <= degree <= maxDegree()`.
    /// @throws std::invalid_argument when `degree` is out of range.
    [[nodiscard]] Eigen::MatrixXcd differential(std::size_t degree) const;

    /// Hodge Laplacian of the product complex,
    /// \f$ \Delta_n = \partial_n^\dagger\partial_n +
    ///     \partial_{n+1}\partial_{n+1}^\dagger \f$
    /// (identity metrics on the factor chain groups).
    [[nodiscard]] Eigen::MatrixXcd laplacian(std::size_t degree) const;

    /// Hodge Laplacian \f$ \Delta_p^A \f$ of factor A alone (same identity
    /// metric convention) — the fixture against which the pairwise-sum
    /// spectral identity is checked.
    [[nodiscard]] Eigen::MatrixXcd factorLaplacianA(std::size_t degree) const;

    /// Hodge Laplacian \f$ \Delta_q^B \f$ of factor B alone.
    [[nodiscard]] Eigen::MatrixXcd factorLaplacianB(std::size_t degree) const;

  private:
    static void validateFactor(const std::vector<std::size_t>& dims,
                               const std::vector<Eigen::MatrixXcd>& diff,
                               double boundaryTolerance, const char* name);
    static Eigen::MatrixXcd factorLaplacian(
        const std::vector<std::size_t>& dims,
        const std::vector<Eigen::MatrixXcd>& diff, std::size_t degree,
        const char* name);
    /// kron(P, Q): entry ((rP·nQ + rQ), (cP·nQ + cQ)) = P(rP,cP)·Q(rQ,cQ).
    static Eigen::MatrixXcd kronDense(const Eigen::MatrixXcd& p,
                                      const Eigen::MatrixXcd& q);

    std::vector<std::size_t> dimsA_{};
    std::vector<Eigen::MatrixXcd> diffA_{};
    std::vector<std::size_t> dimsB_{};
    std::vector<Eigen::MatrixXcd> diffB_{};
};

/// # FockDirectSum
///
/// The Fock direct-sum functor on a bipartition of the modes:
/// \f$ F(h_A \oplus h_B) \cong F(h_A) \otimes F(h_B) \f$ (graded tensor
/// product), compiled with A-modes first. The identification is sign-free
/// on basis states:
/// \f$ |b\rangle \leftrightarrow |b_A\rangle\otimes|b_B\rangle \f$ at joint
/// Fock index \f$ n(b) = i_A + 2^{M_A}\, i_B \f$ — because listing the
/// occupied A-modes before the occupied B-modes is already the wedge word
/// of the joint state.
///
/// Under this identification (all identities exact, integer signs):
///   • even operators lift as \f$ X_A \mapsto X_A\otimes 1 \f$,
///     \f$ Y_B \mapsto 1\otimes Y_B \f$;
///   • odd right-factor operators acquire the parity twist
///     \f$ Y_B \mapsto (-1)^{N_A}\otimes Y_B \f$ (the graded tensor
///     product's Koszul sign — the Jordan-Wigner string over A);
///   • joint CAR generators satisfy `creation(i in A) = liftLeft(a_i†)`,
///     `creation(j in B) = liftRight(a_j†, odd)`, i.e. direct sums become
///     graded tensor products;
///   • \f$ d\Gamma\big(\begin{smallmatrix}L_A & C\\ C^\dagger &
///     L_B\end{smallmatrix}\big) = d\Gamma(L_A)\otimes 1 + 1\otimes
///     d\Gamma(L_B) + \sum_{i\in A,\,j\in B}\big(C_{ij}\,a_i^\dagger a_j +
///     \overline{C_{ij}}\, a_j^\dagger a_i\big) \f$ — coupling blocks become
///     hopping terms;
///   • the graded swap \f$ S(x\otimes y) = (-1)^{|x||y|}\, y\otimes x \f$
///     exchanges the factors with odd/odd sign −1 and +1 on every other
///     elementary parity combination.
///
/// Reference: Jordan & Wigner, "Ueber das Paulische Aequivalenzverbot"
/// (1928), for the parity string carried by the odd lift.
class FockDirectSum {
  public:
    using Complex = std::complex<double>;
    using SparseOp = Eigen::SparseMatrix<Complex>;

    /// Bipartition with `modesA` left modes (joint modes 0..M_A−1) and
    /// `modesB` right modes (joint modes M_A..M_A+M_B−1).
    /// @throws std::invalid_argument if the joint algebra would exceed
    ///         ExteriorAlgebra::kMaxMatrixModes.
    FockDirectSum(std::size_t modesA, std::size_t modesB);

    [[nodiscard]] std::size_t modesA() const noexcept { return modesA_; }
    [[nodiscard]] std::size_t modesB() const noexcept { return modesB_; }

    /// The joint exterior algebra over \f$ M_A + M_B \f$ modes.
    [[nodiscard]] ExteriorAlgebra jointAlgebra() const;
    /// The left-factor algebra over \f$ M_A \f$ modes.
    [[nodiscard]] ExteriorAlgebra leftAlgebra() const;
    /// The right-factor algebra over \f$ M_B \f$ modes.
    [[nodiscard]] ExteriorAlgebra rightAlgebra() const;

    /// Lift a left-factor operator: \f$ X \mapsto X\otimes 1_{F_B} \f$.
    /// Exact for every operator (odd or even) because A-modes precede all
    /// B-modes in the compilation order (empty Jordan-Wigner string).
    /// @throws std::invalid_argument on a dimension mismatch.
    [[nodiscard]] SparseOp liftLeft(const SparseOp& opA) const;

    /// Lift a right-factor operator. `oddOperator == false`:
    /// \f$ Y \mapsto 1_{F_A}\otimes Y \f$. `oddOperator == true` (odd
    /// products of CAR generators): \f$ Y \mapsto (-1)^{N_A}\otimes Y \f$ —
    /// the Koszul/Jordan-Wigner parity twist that makes the lift satisfy
    /// the joint CAR exactly.
    /// @throws std::invalid_argument on a dimension mismatch.
    [[nodiscard]] SparseOp liftRight(const SparseOp& opB,
                                     bool oddOperator) const;

    /// The graded swap \f$ S: F_A\otimes F_B \to F_B\otimes F_A \f$,
    /// \f$ S(x\otimes y) = (-1)^{|x||y|}\, y\otimes x \f$ on homogeneous
    /// parity sectors: row \f$ i_B + 2^{M_B} i_A \f$, column
    /// \f$ i_A + 2^{M_A} i_B \f$, value
    /// \f$ (-1)^{\mathrm{popcount}(i_A)\cdot\mathrm{popcount}(i_B)} \f$.
    /// Odd/odd exchange is −1; every other elementary parity combination
    /// is +1; the reverse swap composes to the identity.
    [[nodiscard]] SparseOp gradedSwapMatrix() const;

    /// Assemble the block one-particle matrix
    /// \f$ L = \begin{pmatrix} L_A & C \\ C^\dagger & L_B \end{pmatrix} \f$
    /// on \f$ h_A \oplus h_B \f$ from an \f$ M_A\times M_A \f$ block, an
    /// \f$ M_B\times M_B \f$ block and an \f$ M_A\times M_B \f$ coupling.
    /// @throws std::invalid_argument on shape mismatches.
    [[nodiscard]] static Eigen::MatrixXcd assembleBlockOneParticle(
        const Eigen::MatrixXcd& blockA, const Eigen::MatrixXcd& blockB,
        const Eigen::MatrixXcd& coupling);

    /// \f$ d\Gamma \f$ of the assembled block one-particle operator on the
    /// joint Fock space — direct sums become graded tensor products and the
    /// coupling block becomes the hopping term (see class docs).
    [[nodiscard]] SparseOp dGammaBlock(const Eigen::MatrixXcd& blockA,
                                       const Eigen::MatrixXcd& blockB,
                                       const Eigen::MatrixXcd& coupling) const;

  private:
    /// Sparse kron(P, Q) with the same index convention as
    /// GradedTensorComplex::kronDense.
    static SparseOp kronSparse(const SparseOp& p, const SparseOp& q);

    std::size_t modesA_{0};
    std::size_t modesB_{0};
};

/// One edge-mode record of an EdgeModeRegistry (edge-mode semantics,
/// minus geometry — the Edge's complex length stays on the Edge and is not
/// duplicated here). Stores the oriented incidence data of the mode's edge
/// and the mode's identity; never a per-edge state vector.
struct EdgeModeRecord {
    /// Stored tail vertex of the edge direction as recorded.
    std::uint64_t vertexA{0};
    /// Stored head vertex of the edge direction as recorded.
    std::uint64_t vertexB{0};
    /// Orientation sign ±1 relative to the stored direction; flips on a
    /// storage reversal so the effective oriented edge is unchanged.
    int orientationSign{+1};
    /// Identity of the two-level mode factor span{|0⟩,|1⟩} this edge indexes
    /// inside the global exterior Fock space.
    std::uint64_t modeId{0};
    /// Oriented component lineage label supplied by the component hierarchy;
    /// the primary key of the deterministic compilation order.
    std::string lineageKey{};
};

/// # EdgeModeRegistry
///
/// The edge-mode basis bookkeeping for the exterior algebra: each edge
/// indexes exactly one two-level mode factor (a modeId) — the registry
/// stores oriented incidence and lineage, never per-edge state vectors, and
/// never touches the Edge's single complex length. A per-edge occupation is
/// a derived marginal of the global state, not a stored product state.
///
/// ## Deterministic compilation order
///
/// `canonicalModeOrder` sorts modes by
/// `(lineageKey, min(vertexA,vertexB), max(vertexA,vertexB))` — oriented
/// component lineage first, the unordered vertex pair as the deterministic
/// tie-break. The order is a compilation artifact of the order-independent
/// abstract exterior algebra (no Kasteleyn orientation is required); it
/// exists so that bitsets and matrices can be built reproducibly.
///
/// ## Relabeling parity
///
/// A vertex relabeling (`relabeled`) rebuilds the canonical order; the
/// induced mode permutation (`orderPermutation`) acts on the Fock space as
/// the exact signed unitary ExteriorAlgebra::modePermutationMatrix, whose
/// per-state sign is OccupationBitset::permutationParity. Every physical
/// amplitude is invariant under relabeling once that parity is applied.
///
/// ## Reorientation convention
///
/// `reverseStoredDirection` swaps the stored endpoints and flips
/// `orientationSign`: the effective oriented edge — and therefore
/// `canonicalOrientationSign`, the compilation order, and every amplitude —
/// is invariant. `flipOrientation` flips only `orientationSign`: the edge is
/// physically reversed, which multiplies the mode's one-particle embedding
/// vector by −1 (a_e ↦ −a_e, a_e† ↦ −a_e†; the two-level factor itself is
/// fixed pointwise and nothing is conjugated), preserving the CAR and every
/// occupation observable while flipping one-particle amplitudes.
class EdgeModeRegistry {
  public:
    EdgeModeRegistry() = default;

    /// Register one two-level occupation mode per edge of `spacetime`, over its
    /// whole edge list. The carrier is \f$ \mathcal{F}(\mathfrak{h}_K) \f$ with
    /// \f$ \mathfrak{h}_K = \operatorname{span}\{|e\rangle : e \in K_1\} \f$;
    /// any per-band carrier is a derived view of these per-edge modes.
    ///
    /// Each edge is registered on its own stored source → target direction with
    /// `orientationSign = +1`, so `canonicalOrientationSign` reports exactly how
    /// that stored orientation sits against the canonical min → max direction.
    /// Every mode gets `lineageKey`, so the canonical order reduces to the
    /// deterministic endpoint sort; a caller with genuine component lineage
    /// assigns it afterwards. Edges with a missing endpoint are skipped. The
    /// registry stores incidence and lineage only — it never reads a length or a
    /// connection phase.
    ///
    /// @throws std::invalid_argument on a self-loop or a duplicated unordered
    ///         vertex pair, both of which are malformed in a simplicial complex.
    [[nodiscard]] static EdgeModeRegistry fromSpacetime(
        const ::tessera::spacetime::Spacetime& spacetime,
        std::string lineageKey = "K1");

    /// Register the edge (vertexA → vertexB) with orientation sign ±1 and
    /// its oriented-component lineage key. Returns the assigned modeId
    /// (dense, in registration order — registration order carries no
    /// physical meaning; only the canonical order does).
    /// @throws std::invalid_argument on a self-loop, a sign outside {−1,+1},
    ///         or a duplicate unordered vertex pair.
    std::uint64_t addEdge(std::uint64_t vertexA, std::uint64_t vertexB,
                          int orientationSign, std::string lineageKey);

    /// Number of registered edge modes.
    [[nodiscard]] std::size_t modeCount() const noexcept {
        return records_.size();
    }

    /// The record for `modeId`.
    /// @throws std::invalid_argument on an unknown modeId.
    [[nodiscard]] const EdgeModeRecord& record(std::uint64_t modeId) const;

    /// All records, indexed by modeId.
    [[nodiscard]] const std::vector<EdgeModeRecord>& records() const noexcept {
        return records_;
    }

    /// Reverse the stored direction of `modeId`: (a, b, s) → (b, a, −s).
    /// A pure storage convention change — the effective oriented edge, the
    /// canonical order and all amplitudes are unchanged.
    void reverseStoredDirection(std::uint64_t modeId);

    /// Physically reverse the oriented edge of `modeId`: s → −s with the
    /// stored direction fixed. One-particle amplitudes attached to this mode
    /// flip sign; occupation observables are unchanged.
    void flipOrientation(std::uint64_t modeId);

    /// The orientation of `modeId` relative to the canonical
    /// (min-vertex → max-vertex) direction:
    /// `orientationSign × (+1 if vertexA < vertexB else −1)`.
    /// Invariant under `reverseStoredDirection`; flips under
    /// `flipOrientation`.
    [[nodiscard]] int canonicalOrientationSign(std::uint64_t modeId) const;

    /// The deterministic compilation order: modeIds sorted by
    /// `(lineageKey, min(vA,vB), max(vA,vB))`.
    [[nodiscard]] std::vector<std::uint64_t> canonicalModeOrder() const;

    /// `positions[modeId]` = index of `modeId` in `canonicalModeOrder()`.
    [[nodiscard]] std::vector<std::size_t> compilationPositions() const;

    /// The registry after the vertex relabeling `vertexMap` (must cover
    /// every vertex used and be injective on them). modeIds, orientation
    /// signs and lineage keys are preserved; only vertex ids change — the
    /// canonical order is then rebuilt from the new ids.
    /// @throws std::invalid_argument on a missing or non-injective mapping.
    [[nodiscard]] EdgeModeRegistry relabeled(
        const std::unordered_map<std::uint64_t, std::uint64_t>& vertexMap)
        const;

    /// The mode permutation induced by rebuilding the canonical order:
    /// `perm[i]` is the position in `after`'s canonical order of the mode
    /// sitting at position `i` of `before`'s canonical order (matched by
    /// modeId). Feed to OccupationBitset::permutationParity /
    /// ExteriorAlgebra::modePermutationMatrix for the exact parity map.
    /// @throws std::invalid_argument when the registries do not hold the
    ///         same modeId set.
    [[nodiscard]] static std::vector<std::size_t> orderPermutation(
        const EdgeModeRegistry& before, const EdgeModeRegistry& after);

  private:
    void validateModeId(std::uint64_t modeId) const;

    std::vector<EdgeModeRecord> records_{};
};

}  // namespace tessera::quantum
