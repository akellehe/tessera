// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_CHAINHODGE_PENCILSCHUR_H
#define TESSERA_CHAINHODGE_PENCILSCHUR_H

#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "chainhodge/ChainHodge.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"

namespace tessera::chainhodge {

/// One Feshbach complement of a symmetric pencil, with the projectors and the
/// residuals that certify it.
///
/// Away from an interior resonance the complement is the exact
/// \f$ F_B(\lambda) \f$ and `interiorSingular` is false. At an interior
/// resonance (\f$ P_{II} \f$ singular at the shift) the inverse is replaced by
/// the declared supported generalized inverse \f$ P_{II}^{\#} \f$ — the
/// Moore–Penrose pseudoinverse taken from the singular value decomposition at
/// the declared rank tolerance — only after the compatibility condition has
/// been measured, and the resonant interior modes are retained explicitly as
/// fiber coordinates rather than eliminated.
struct FeshbachResult {
  Complex lambda{0.0, 0.0};
  /// Interface (kept) and interior (eliminated) coordinates, ascending.
  std::vector<int> interface{};
  std::vector<int> interior{};
  /// \f$ F_B(\lambda) = P_{BB} - P_{BI} P_{II}^{-1} P_{IB} \f$, \f$ P = A - \lambda M \f$.
  /// At an interior resonance the inverse is the generalized inverse
  /// \f$ P_{II}^{\#} \f$ and this is the block of `resonantResponse` on the
  /// interface coordinates alone.
  Eigen::MatrixXcd response{};
  /// The constraint modes \f$ T = [I_B;\ -P_{II}^{-1} P_{IB}] \f$ in the full
  /// coordinate order (\f$ n \times |B| \f$): the fibers the kept coordinates
  /// carry, whose congruence \f$ T^T M T \f$ is the inherited chain metric.
  /// At an interior resonance the inverse is \f$ P_{II}^{\#} \f$.
  Eigen::MatrixXcd constraintModes{};
  Complex interiorDeterminant{0.0, 0.0};
  Complex responseDeterminant{0.0, 0.0};
  Complex pencilDeterminant{0.0, 0.0};
  /// \f$ |\det P - \det P_{II}\det F_B| / \max(|\det P|, \epsilon) \f$. Quiet
  /// NaN at an interior resonance, where the factorization has no content
  /// (both sides vanish).
  double determinantResidual{std::numeric_limits<double>::quiet_NaN()};
  /// Relative residual of the interior solve.
  double solveResidual{std::numeric_limits<double>::quiet_NaN()};
  /// True when \f$ P_{II} \f$ was singular at the shift: an interior
  /// resonance, where the fields below carry the resonant reduction.
  bool interiorSingular{false};

  // --- the generalized inverse, its projectors, and the resonant reduction ---

  /// Numerical rank of \f$ P_{II} \f$ and the absolute singular-value threshold
  /// that decided it, \f$ \text{rankTolerance}\cdot\varsigma_{\max}(P_{II}) \f$.
  /// The singular value decomposition that measures them is taken only at a
  /// resonance, where the generalized inverse needs it; away from one the
  /// pivoted LU declares the block invertible and the rank is \f$ |I| \f$ with
  /// threshold zero.
  int interiorRank{0};
  double interiorRankThreshold{0.0};
  /// \f$ \varsigma_r/\varsigma_{r+1} \f$ of \f$ P_{II} \f$: the last kept over
  /// the first discarded singular value, the margin the resonance was declared
  /// with. \f$ +\infty \f$ away from a resonance and when nothing was
  /// discarded.
  double interiorSingularGap{std::numeric_limits<double>::infinity()};
  /// A basis of \f$ \ker P_{II} \f$ (\f$ |I| \times q \f$, orthonormal columns):
  /// the resonant interior modes, empty away from a resonance.
  Eigen::MatrixXcd interiorNullSpace{};
  /// A basis of \f$ \ker P_{II}^T \f$ (\f$ |I| \times q \f$, orthonormal
  /// columns): the left null space in the transpose pairing the theory uses.
  Eigen::MatrixXcd interiorLeftNullSpace{};
  /// The resonant modes in the full coordinate order, \f$ [0;\ N] \f$
  /// (\f$ n \times q \f$): the fiber coordinates the retained interior modes
  /// carry, alongside `constraintModes`.
  Eigen::MatrixXcd resonantModes{};
  /// \f$ \Pi_R = P_{II}P_{II}^{\#} \f$, the projector onto
  /// \f$ \operatorname{ran} P_{II} \f$, and \f$ \Pi_N = I - P_{II}^{\#}P_{II} \f$,
  /// the projector onto \f$ \ker P_{II} \f$. Both are recorded at every shift,
  /// resonant or not (away from a resonance \f$ \Pi_R = I \f$, \f$ \Pi_N = 0 \f$).
  Eigen::MatrixXcd rangeProjector{};
  Eigen::MatrixXcd nullProjector{};
  /// The compatibility (solvability) condition of the whitepaper,
  /// \f$ y^T P_{IB} x_B = 0 \f$ for every \f$ y \in \ker P_{II}^T \f$, measured
  /// as \f$ \|N_L^T P_{IB}\| / \|P_{IB}\| \f$ with \f$ N_L \f$ the columns of
  /// `interiorLeftNullSpace`. Zero exactly when \f$ P_{IB} \f$ maps the
  /// interface into \f$ \operatorname{ran} P_{II} \f$, which is when the
  /// interior equation is solvable for every interface load. Zero away from a
  /// resonance, where the left null space is empty.
  double compatibilityResidual{0.0};
  /// Whether `compatibilityResidual` is at or below the declared rank
  /// tolerance: the condition the inverse may be replaced under.
  bool compatible{true};
  /// The independence condition of the whitepaper,
  /// \f$ P_{BI}\ker P_{II} = 0 \f$, measured as \f$ \|P_{BI}N\| / \|P_{BI}\| \f$
  /// with \f$ N \f$ the columns of `interiorNullSpace`. When it holds the
  /// boundary response does not depend on which interior solution was chosen
  /// and `response` alone is the reduction; otherwise the null modes are
  /// carried as the extra coordinates of `resonantResponse`.
  double independenceResidual{0.0};
  bool responseIndependent{true};
  /// The resonant reduction over the retained coordinates \f$ (x_B, c) \f$ —
  /// the interface coordinates followed by the \f$ q \f$ resonant mode
  /// amplitudes — as the square \f$ (|B|+q) \times (|B|+q) \f$ matrix
  /// \f[
  ///   \hat F(\lambda) = \begin{pmatrix}
  ///     P_{BB} - P_{BI}P_{II}^{\#}P_{IB} & P_{BI}N \\
  ///     N_L^T P_{IB} & 0 \end{pmatrix},
  /// \f]
  /// whose first block row is the interface equation after the interior has
  /// been eliminated on \f$ \operatorname{ran} P_{II} \f$ and whose second is
  /// the compatibility constraint. Empty away from a resonance, where
  /// `response` is the whole reduction.
  Eigen::MatrixXcd resonantResponse{};
  /// The numerical certificate of the resonant reduction as a whole, measured
  /// on every retained coordinate rather than only on the null space. With
  /// \f$ W = [\,T \mid Z\,] \f$ the retained fibers — the constraint modes
  /// beside the resonant modes — the reduction is exact in the sense that
  /// \f[
  ///   P(\lambda)\,W = E_B\,\hat F(\lambda)_{[1..|B|]}
  ///                 + E_I\,\overline{N_L}\,\hat F(\lambda)_{[|B|+1..]},
  /// \f]
  /// where \f$ E_B \f$ and \f$ E_I \f$ inject the interface and interior
  /// coordinates and \f$ \overline{N_L} \f$ is the conjugate of
  /// `interiorLeftNullSpace` (a basis of \f$ \ker P_{II}^H \f$, the complement
  /// of \f$ \operatorname{ran}P_{II} \f$). Every column of the pencil applied
  /// to a retained fiber is therefore read off \f$ \hat F(\lambda) \f$ alone,
  /// and this is the relative Frobenius residual of that identity.
  double reductionResidual{std::numeric_limits<double>::quiet_NaN()};
  /// The numerical certificate of the resonant reduction: every null vector
  /// \f$ (x_B, c) \f$ of \f$ \hat F(\lambda) \f$ lifts to the null vector
  /// \f$ x = [x_B;\ -P_{II}^{\#}P_{IB}x_B + Nc] \f$ of \f$ P(\lambda) \f$, and
  /// this is \f$ \max \|P x\| / (\|P\|\,\|x\|) \f$ over a basis of
  /// \f$ \ker\hat F(\lambda) \f$ taken at the declared rank tolerance. Quiet
  /// NaN away from a resonance and when \f$ \hat F(\lambda) \f$ is nonsingular
  /// (there is then no null vector to lift, and \f$ \lambda \f$ is not an
  /// eigenvalue of the pencil).
  double liftResidual{std::numeric_limits<double>::quiet_NaN()};
};

/// A congruence \f$ (T^T A T,\ T^T M T) \f$ with the residuals that say how far
/// the reduced pair is from the pair it was built from.
struct CongruenceResult {
  Eigen::MatrixXcd A{};
  Eigen::MatrixXcd M{};
  /// \f$ \|(T^TAT)^T - T^TAT\| / \|T^TAT\| \f$ and the same for \f$ T^TMT \f$:
  /// a congruence of a complex symmetric pair is complex symmetric, so these
  /// measure whether the pair that went in was symmetric in the transpose
  /// pairing rather than assuming it. Quiet NaN for an empty congruence.
  double symmetryDefect{std::numeric_limits<double>::quiet_NaN()};
  double metricSymmetryDefect{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ \varsigma_{\min}(T)/\varsigma_{\max}(T) \f$: how independent the
  /// columns of the reduction basis are. Zero when the basis is rank deficient,
  /// in which case the reduced pair is singular and its spectrum is not the
  /// surrogate spectrum. Quiet NaN for an empty basis.
  double basisConditionInverse{std::numeric_limits<double>::quiet_NaN()};
};

/// # SurrogateResult
///
/// A certified Craig–Bampton / AMLS surrogate of the pencil
/// \f$ \mathcal P(\lambda) = A - \lambda M \f$ over a declared complex spectral
/// window, and the numerical certificate that its spectrum sits on the exact
/// Feshbach map.
///
/// The reduction basis is \f$ V = [\,T \mid \Psi\,] \f$ with \f$ T \f$ the
/// interface constraint modes at the declared shift \f$ \lambda_0 \f$ (the
/// `constraintModes` of `PencilSchur::feshbach`) and \f$ \Psi = [0;\ U] \f$ the
/// retained fixed-interface modes: the eigenvectors of the interior pencil
/// \f$ (A_{II}, M_{II}) \f$ whose eigenvalues lie in the retention disc. The
/// reduced pair is the congruence \f$ (V^TAV,\ V^TMV) \f$ — the transpose
/// pairing throughout, so the construction is the same in the complex
/// symmetric and the non-normal regime and no Hermitian form is used anywhere.
///
/// The window is a closed disc \f$ |\theta - c| \le \rho \f$ in the complex
/// spectral plane, because a non-normal or complex symmetric pencil has a
/// complex spectrum and an interval on the real axis would not enclose it; the
/// disc is the same object `Contour::circle` integrates over. A fixed-interface
/// mode is retained when \f$ |\theta - c| \le \rho_{\text{cut}} \f$ for the
/// declared retention radius \f$ \rho_{\text{cut}} \f$.
///
/// The certificate is an inequality that holds exactly, verified numerically on
/// every retained pair. For a surrogate pair \f$ (\theta, y) \f$ with fine
/// vector \f$ x = Vy \f$ and \f$ P = \mathcal P(\theta) \f$, splitting \f$ x \f$
/// into its interface part \f$ x_B \f$ and interior part \f$ x_I \f$ gives
/// \f[
///   F_B(\theta)\,x_B = (Px)_B - P_{BI}P_{II}^{-1}(Px)_I,
/// \f]
/// so that \f$ \|F_B(\theta)x_B\| \le (1 + \|P_{BI}P_{II}^{-1}\|)\,\|Px\| \f$.
/// The left-hand side, normalized, is `feshbachDefects`; the right-hand side,
/// normalized the same way, is `feshbachBounds`. An exact eigenpair has
/// \f$ Px = 0 \f$ and therefore \f$ F_B(\theta)x_B = 0 \f$: the surrogate
/// eigenvalue is held to the exact Feshbach map to within its own residual,
/// and how far it is held is the bound, not an assertion.
struct SurrogateResult {
  /// Interface (retained) and interior (reduced) coordinates, ascending.
  std::vector<int> interface{};
  std::vector<int> interior{};
  /// The shift \f$ \lambda_0 \f$ the interface constraint modes were taken at.
  Complex shift{0.0, 0.0};
  /// Centre \f$ c \f$ and radius \f$ \rho \f$ of the declared window disc, and
  /// the retention radius \f$ \rho_{\text{cut}} \f$ of the fixed-interface
  /// modes. A retention radius below \f$ \rho \f$ discards modes from inside
  /// the window, which is allowed and reported — `discardedModeSeparation`
  /// turns negative and the surrogate does not certify — rather than refused,
  /// because the surrogate's numbers are worth reading either way.
  Complex windowCentre{0.0, 0.0};
  double windowRadius{0.0};
  double retentionRadius{0.0};
  /// The reduction basis \f$ V \f$ (\f$ n \times (|B| + m) \f$) and the reduced
  /// pair \f$ (\hat A, \hat M) = (V^TAV,\ V^TMV) \f$ with its congruence
  /// residuals.
  Eigen::MatrixXcd basis{};
  CongruenceResult reduced{};
  /// The whole fixed-interface spectrum of \f$ (A_{II}, M_{II}) \f$, sorted by
  /// \f$ (\mathrm{Re},\mathrm{Im}) \f$, and how many of its modes were retained.
  std::vector<Complex> interiorEigenvalues{};
  int retainedModes{0};
  /// \f$ \min_{\text{discarded}} |\theta - c| - \rho \f$: the separation of the
  /// nearest discarded fixed-interface mode from the window, the quantity the
  /// truncation error is controlled by. \f$ +\infty \f$ when nothing was
  /// discarded; negative when a discarded mode lies inside the window, which
  /// makes the surrogate uncertified.
  double discardedModeSeparation{std::numeric_limits<double>::infinity()};
  /// The reduced spectrum, sorted by \f$ (\mathrm{Re},\mathrm{Im}) \f$, and its
  /// right eigenvectors as columns in the same order.
  std::vector<Complex> eigenvalues{};
  Eigen::MatrixXcd vectors{};
  /// The indices, into `eigenvalues`, of the reduced eigenvalues that lie in
  /// the window disc: the ones the surrogate claims. The per-pair vectors below
  /// are indexed in this same order.
  std::vector<int> windowIndices{};
  /// \f$ \|\mathcal P(\theta)x\| / (\|A\|\,\|x\|) \f$ with \f$ x = Vy \f$, the
  /// fine-space residual of each claimed pair.
  std::vector<double> residuals{};
  /// \f$ \|F_B(\theta)x_B\| / (\|F_B(\theta)\|\,\|x_B\|) \f$, the distance of
  /// the claimed pair from the exact Feshbach map.
  std::vector<double> feshbachDefects{};
  /// \f$ (1 + \|P_{BI}P_{II}^{-1}\|)\,\|\mathcal P(\theta)x\| /
  /// (\|F_B(\theta)\|\,\|x_B\|) \f$, the certified bound the defect is held
  /// within.
  std::vector<double> feshbachBounds{};
  /// Whether each claimed pair's defect is within its bound, measured rather
  /// than assumed.
  std::vector<bool> feshbachHolds{};
  /// Whether \f$ \theta \f$ was an interior resonance of the pencil, where the
  /// bound above is not defined because \f$ P_{II}^{-1} \f$ is not; the defect
  /// is then measured against the resonant reduction \f$ \hat F(\theta) \f$
  /// instead and the bound is quiet NaN.
  std::vector<bool> resonantAtEigenvalue{};
  /// The declared acceptance tolerance the certificate holds against.
  double tolerance{0.0};
  /// The whole surrogate's verdict: every claimed pair's defect is within its
  /// certified bound, every bound is at or below `tolerance`, and no discarded
  /// fixed-interface mode lies inside the window. A surrogate that fails is
  /// returned with its numbers rather than refused.
  bool certified{false};
  /// Why it failed, by name; empty when `certified`.
  std::string refusal{};
};

/// The coarse pencil and chain metric restricted to retained fibers,
/// \f$ (\hat A, \mathcal G) = (Z^T \tilde A Z,\ Z^T M Z) \f$, with the block
/// offsets of the fibers that were concatenated.
struct FiberRestriction {
  Eigen::MatrixXcd A{};
  Eigen::MatrixXcd gram{};
  std::vector<int> blockOffsets{};
  std::vector<int> blockRanks{};
};

/// A transfer between two fibers with its reversal certificate.
struct TransferResult {
  /// \f$ T_{AB}(U) = (Z_A^\vee)^T (\tilde A^U)_{AB} Z_B \f$.
  Eigen::MatrixXcd forward{};
  /// \f$ T_{BA}(U^{-1}) = (Z_B^\vee(U^{-1}))^T (\tilde A^{U^{-1}})_{BA} Z_A(U^{-1})
  ///   = Z_B^T \tilde A^{U^{-1}} Z_A^\vee \f$.
  Eigen::MatrixXcd reverse{};
  /// \f$ \|T_{BA}(U^{-1}) - T_{AB}(U)^T\| / \|T_{AB}\| \f$: the reversal identity.
  double reversalResidual{std::numeric_limits<double>::quiet_NaN()};
  double tolerance{0.0};
  /// The groupoid hypothesis \f$ T_{BA} = T_{AB}^{-1} \f$, measured as
  /// \f$ \|T_{BA} T_{AB} - I\| \f$ (square, same rank); false otherwise.
  bool groupoidHolds{false};
  double groupoidResidual{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ T_{AB}^{-T} \f$, the dual transfer \f$ M^\vee = M^{-T} \f$, emitted only
  /// when the groupoid hypothesis holds; empty otherwise.
  Eigen::MatrixXcd dualTransfer{};
};

/// # PencilSchur
///
/// The recursion on the symmetric pencil \f$ \mathcal P(\lambda) = \tilde A -
/// \lambda M \f$ on geometric images, dense below the crossover. Partition
/// coordinates into interface \f$ B \f$ and interior \f$ I \f$.
///
/// * (a) `feshbach`: \f$ F_B(\lambda) = \mathcal P_{BB} - \mathcal P_{BI}\mathcal P_{II}^{-1}
///   \mathcal P_{IB} \f$, \f$ \det\mathcal P = \det\mathcal P_{II}\det F_B \f$;
///   \f$ F_B(\lambda;U)^T = F_B(\lambda;U^{-1}) \f$, symmetric at \f$ U = 1 \f$.
/// * (b) `craigBampton`: the congruence \f$ (T^T\tilde A T,\ T^T M T) \f$ of an
///   explicit basis, and the overload that builds the certified
///   Craig–Bampton/AMLS surrogate over a declared window disc itself and holds
///   its spectrum to the exact Feshbach map (`SurrogateResult`).
/// * (c) `restrictToFibers`: retained fibers with images \f$ Z \f$ give
///   \f$ \mathcal G_{\ell+1} = Z^T M Z \f$ and \f$ \hat A_{\ell+1} = Z^T\tilde A Z \f$;
///   off-diagonal blocks \f$ Z_A^T M Z_B \f$ vanish unless the supports of
///   the two images share a top simplex (`supportsShareTopSimplex`).
/// * (d) `transfer`: \f$ T_{AB}(U) = (Z_A^\vee)^T(\tilde A^U)_{AB}Z_B \f$ with
///   \f$ T_{BA}(U^{-1}) = T_{AB}(U)^T \f$ enforced as a runtime assertion, and
///   \f$ M^\vee = M^{-T} \f$ only under the certified groupoid hypothesis.
///
/// Every pairing is the transpose; no conjugation enters, so nothing here
/// distinguishes the complex symmetric regime from the non-normal one: both
/// run the same code, and the regime is a measured property of the operands
/// (`CovariantChainHodge::regimeCertificate`), never a branch.
class PencilSchur {
 public:
  /// The Feshbach complement at \p lambda over the kept coordinates
  /// \p interface, with the projectors of the interior block and, at an
  /// interior resonance, the generalized inverse, the compatibility and
  /// independence residuals, the retained resonant modes, the resonant
  /// reduction \f$ \hat F(\lambda) \f$ and the lift residual that certifies it
  /// (see `FeshbachResult`).
  /// @param rankTolerance relative singular-value threshold deciding the rank
  ///   of \f$ P_{II} \f$: a singular value at or below
  ///   \p rankTolerance \f$ \cdot\,\varsigma_{\max}(P_{II}) \f$ is zero, and
  ///   \f$ \lambda \f$ is an interior resonance.
  /// @throws std::invalid_argument when \p A and \p M are not square of the
  ///   same size or an interface index is out of range.
  [[nodiscard]] static FeshbachResult feshbach(const Eigen::MatrixXcd &A,
                                               const Eigen::MatrixXcd &M, Complex lambda,
                                               const std::vector<int> &interface,
                                               double rankTolerance = 1e-12);
  /// The same Feshbach complement on the sparse production path: \f$ A \f$ and
  /// \f$ M \f$ are sparse, the interior block is factorized by sparse LU, and
  /// the only dense object formed is the \f$ n \times |B| \f$ block of
  /// constraint modes and the \f$ |B| \times |B| \f$ response. No
  /// \f$ n \times n \f$ matrix appears, so this is defined at and above the
  /// crossover where the dense reading refuses.
  ///
  /// The projectors, the generalized inverse and the resonant reduction are not
  /// available here: they rest on a singular value decomposition of the
  /// interior block, which is dense. An interior resonance is therefore
  /// detected — the sparse factorization of \f$ P_{II} \f$ fails, or its
  /// determinant is below \p rankTolerance times the largest modulus on its
  /// diagonal — and refused by name, with the dense reading named as the one
  /// that resolves it.
  /// @param report when non-null, receives the cost of the interior
  ///   factorization: its wall time, the memory of its factors and their
  ///   fill-in.
  /// @throws std::invalid_argument when \p A and \p M are not square of the
  ///   same size or an interface index is out of range; std::runtime_error at
  ///   an interior resonance.
  [[nodiscard]] static FeshbachResult sparseFeshbach(const SparseMatrix &A, const SparseMatrix &M,
                                                     Complex lambda,
                                                     const std::vector<int> &interface,
                                                     double rankTolerance = 1e-12,
                                                     SparseCostReport *report = nullptr);
  /// The congruence \f$ (T^TAT,\ T^TMT) \f$ of an explicit reduction basis
  /// \p T, with the symmetry defects of the reduced pair and the inverse
  /// condition number of the basis.
  /// @throws std::invalid_argument when \p T does not have as many rows as
  ///   \p A and \p M, which must be square of the same size.
  [[nodiscard]] static CongruenceResult craigBampton(const Eigen::MatrixXcd &A,
                                                     const Eigen::MatrixXcd &M,
                                                     const Eigen::MatrixXcd &T);
  /// The certified Craig–Bampton / AMLS surrogate of \f$ (A, M) \f$ over the
  /// window disc \f$ |\theta - \f$ \p windowCentre \f$ | \le \f$
  /// \p windowRadius, retaining the fixed-interface modes within
  /// \p retentionRadius of the centre, with the interface constraint modes
  /// taken at \p shift. Every claimed eigenvalue is held to the exact Feshbach
  /// map and the inequality that holds it is verified numerically
  /// (see `SurrogateResult`).
  ///
  /// The surrogate runs in every pencil regime: nothing in it forms an adjoint
  /// or assumes a definite form, so the non-normal and the complex symmetric
  /// pencil take the same path, and a pencil that happens to be Hermitian takes
  /// it too.
  /// @param tolerance the declared acceptance tolerance: `certified` requires
  ///   every certified bound to be at or below it.
  /// @param rankTolerance relative rank threshold, as `feshbach`.
  /// @throws std::invalid_argument when \p A and \p M are not square of the
  ///   same size, an interface index is out of range, or either radius is
  ///   negative; std::runtime_error when the interior metric \f$ M_{II} \f$ or the
  ///   reduced metric \f$ V^TMV \f$ is singular, by name, so that a meaningless
  ///   surrogate spectrum is never returned.
  [[nodiscard]] static SurrogateResult craigBampton(const Eigen::MatrixXcd &A,
                                                    const Eigen::MatrixXcd &M,
                                                    const std::vector<int> &interface,
                                                    Complex windowCentre, double windowRadius,
                                                    double retentionRadius,
                                                    Complex shift = Complex(0.0, 0.0),
                                                    double tolerance = 1e-8,
                                                    double rankTolerance = 1e-12);
  /// Concatenated fibers \p Z (columns) as one block.
  [[nodiscard]] static FiberRestriction restrictToFibers(const Eigen::MatrixXcd &A,
                                                         const Eigen::MatrixXcd &M,
                                                         const Eigen::MatrixXcd &Z);
  /// Several fibers, each a block of columns, with block offsets recorded.
  [[nodiscard]] static FiberRestriction restrictToFibers(
      const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
      const std::vector<Eigen::MatrixXcd> &fibers);
  /// \f$ Z_A^T M Z_B \f$, one off-diagonal Gram block.
  [[nodiscard]] static Eigen::MatrixXcd gramBlock(const Eigen::MatrixXcd &M,
                                                  const Eigen::MatrixXcd &ZA,
                                                  const Eigen::MatrixXcd &ZB);
  /// Whether two image supports (canonical degree-\p k cell indices) share a
  /// top simplex of \p K — the locality condition under which their Gram
  /// block may be nonzero.
  [[nodiscard]] static bool supportsShareTopSimplex(const cobordism::ChainComplex &K, int k,
                                                    const std::vector<int> &supportA,
                                                    const std::vector<int> &supportB);
  /// The support of an image: the indices of its nonzero rows above
  /// \p threshold times its largest modulus.
  [[nodiscard]] static std::vector<int> support(const Eigen::MatrixXcd &Z,
                                                double threshold = 1e-12);
  /// The transfer between fibers with the reversal identity asserted.
  /// @param AtildeU the dressed pencil operator for \f$ U \f$.
  /// @param AtildeUinv the dressed pencil operator for \f$ U^{-1} \f$ (the dual
  ///   instance).
  /// @param ZA the band's images on \f$ A \f$ for \f$ U \f$.
  /// @param ZAdual the band's images on \f$ A \f$ for \f$ U^{-1} \f$.
  /// @param ZB the band's images on \f$ B \f$ for \f$ U \f$.
  /// @param ZBdual the band's images on \f$ B \f$ for \f$ U^{-1} \f$.
  /// @param tolerance relative tolerance on the reversal identity.
  /// @throws std::runtime_error, by name with the measured residual, when
  ///   \f$ \|T_{BA}(U^{-1}) - T_{AB}(U)^T\| \f$ exceeds \p tolerance times
  ///   \f$ \|T_{AB}\| \f$.
  [[nodiscard]] static TransferResult transfer(const Eigen::MatrixXcd &AtildeU,
                                               const Eigen::MatrixXcd &AtildeUinv,
                                               const Eigen::MatrixXcd &ZA,
                                               const Eigen::MatrixXcd &ZAdual,
                                               const Eigen::MatrixXcd &ZB,
                                               const Eigen::MatrixXcd &ZBdual,
                                               double tolerance = 1e-8);
};

}  // namespace tessera::chainhodge

#endif  // TESSERA_CHAINHODGE_PENCILSCHUR_H
