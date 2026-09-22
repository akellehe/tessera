// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_RECURSIVEQUOTIENT_H
#define TESSERA_COBORDISM_RECURSIVEQUOTIENT_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <limits>
#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cobordism/Certificate.h"
#include "cobordism/HodgeLaplacian.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

class AnalyticCache;

/// The declared treatment of the labeled-sum embedding Gram matrix. A run
/// proceeds by exactly one option; independence of the retained fibers' images
/// inside the chain space is never assumed.
enum class FiberEmbeddingPolicy {
  /// Carry \f$ G = J^\dagger W J \f$ exactly in every subsequent formula.
  CarryGramExactly,
  /// Certify \f$ \|G - I\| \le \varepsilon \f$ and propagate
  /// \f$ \varepsilon \f$ through the composable amplitude budget.
  CertifiedNearIsometry,
  /// Quotient \f$ \ker G \f$ and restate the retained ranks.
  QuotientKernel,
};

/// How a level's operator was produced from its parent: the response step
/// \f$ \RN_{\ell+1}(\lambda) = \mathrm{Feshbach}_{P_\ell}(\RN_\ell(\lambda)) \f$.
/// The recursion is over pencils; the static \f$ \lambda = 0 \f$ complement
/// is one point of it.
enum class LevelOrigin {
  /// A base instance built directly over a complex or an explicit matrix.
  Base,
  /// The exact supported static Schur complement at \f$ \lambda = 0 \f$.
  StaticResponse,
  /// The exact energy-dependent Feshbach--Schur pencil evaluated at a
  /// declared \f$ \lambda \f$ over a declared band window.
  BandPencil,
  /// A cached linear Craig--Bampton/AMLS surrogate over a declared frequency
  /// window: a certified approximation, not an exact spectral identity.
  Surrogate,
};

/// Why a retained coordinate was kept instead of eliminated. Harmonic,
/// resonant and selected interior coordinates become explicit stalk
/// coordinates.
enum class RetainedCoordinateKind {
  /// An interface cell (always retained; the \f$ B \f$ block).
  Interface,
  /// An interior kernel mode of \f$ L_{II} \f$ (a topological/harmonic zero
  /// mode at \f$ \lambda = 0 \f$).
  Harmonic,
  /// An interior kernel mode of \f$ L_{II} - \lambda I \f$ at a declared
  /// resonance \f$ \lambda \neq 0 \f$.
  Resonant,
  /// A caller-selected interior cell coordinate.
  Selected,
};

/// # RecursiveQuotient
///
/// Recursive static and shifted response reduction of a (Hodge) operator over
/// a declared cell partition.
///
/// Reference: Horak and Jost, "Spectra of combinatorial Laplace operators on
/// simplicial complexes", arXiv:1105.2712
///
/// ## Exact identities and their domains
///
/// Cells split into interface cells \f$ B \f$ and per-component interior
/// cells \f$ I = \sqcup_v I_v \f$, blocking the operator as
/// \f$ L = \begin{pmatrix} L_{BB} & L_{BI} \\ L_{IB} & L_{II} \end{pmatrix} \f$
/// with \f$ L_{II} \f$ block-diagonal over components.
///
///  - **Static (\f$ \lambda = 0 \f$).** The supported static response
///    \f$ L_{\text{eff}} = L_{BB} - L_{BI} L_{II}^{+} L_{IB} \f$, from factor
///    solves of \f$ L_{II} X = L_{IB} \f$; the inverse is never formed. In the
///    **positive self-adjoint** regime it is the exact interior minimization:
///    for a compatible interface probe \f$ b \f$,
///    \f$ \min_{x_I} [b;x_I]^\dagger L [b;x_I] = b^\dagger L_{\text{eff}} b \f$
///    with minimizer \f$ x_I^* = -L_{II}^{+} L_{IB} b \f$; in the
///    **Hermitian-indefinite** regime a stationarity condition; in the
///    **non-normal** regime certified block elimination, solvable when
///    \f$ L_{IB} b \perp \ker L_{II}^{\dagger} \f$. Interior kernel modes are
///    retained as explicit stalk coordinates.
///  - **Shifted / Feshbach--Schur (band window).** For \f$ \lambda \f$ with
///    \f$ L_{II} - \lambda I \f$ invertible, \f$ F_B(\lambda) = L_{BB} -
///    \lambda I - L_{BI} (L_{II} - \lambda I)^{-1} L_{IB} \f$, with
///    \f$ \det(L - \lambda I) = \det(L_{II} - \lambda I)\det F_B(\lambda) \f$;
///    hence \f$ \lambda \in \operatorname{spec} L \iff 0 \in
///    \operatorname{spec} F_B(\lambda) \f$ away from the interior spectrum.
///    At an interior resonance the resonant modes are retained explicitly,
///    after the check
///    \f$ L_{IB} b \perp \ker (L_{II} - \lambda I)^{\dagger} \f$. The static
///    complement does not preserve the nonzero spectrum, so
///    `Certificate::domain()` distinguishes `Static` from `BandWindow`.
///  - **Surrogate, fiber sum, Fock stage.** `craigBampton` builds a certified
///    Craig--Bampton/AMLS approximation over a declared window;
///    `labeledFiberSum` and `certifiedFiberSum` build the abstract labeled sum
///    \f$ \boxplus_v E_v \f$, with embedding \f$ J \f$ into the chain space
///    and Gram \f$ G = J^\dagger W J \f$ (\f$ J^T M J \f$ on a pencil level,
///    \f$ \tilde Y^T Y \f$ against the bands' own left frames); `fockStage`
///    the free many-body spectrum over that sum, compressed in the same
///    pairing. Fibers may overlap on shared interface cells, so
///    an internal direct sum is never asserted: each run proceeds by one
///    declared `FiberEmbeddingPolicy`.
///  - **Recursion.**
///    \f$ \RN_{\ell+1}(\lambda) = \mathrm{Feshbach}_{P_\ell}(\RN_\ell(\lambda)) \f$,
///    with `childPersistentPartition` supplying \f$ P_\ell \f$ at every scale.
///    The next level is an operator-valued response network: vertices carry
///    the retained fibers, links the effective blocks of the reduced
///    operator.
///
/// ## Metric regimes
///
/// The operator travels with the diagonal chain-space metric \f$ W \f$ it is
/// self-adjoint against (identity unless stated); the regime on every
/// certificate is detected against that metric.
///
///  - `PositiveSemidefinite` — \f$ WL \f$ Hermitian, \f$ W > 0 \f$ and
///    \f$ WL \succeq 0 \f$, verified by a pivoted LDLT below the dense
///    crossover. Energy \f$ x^\dagger W L x \f$ is minimized.
///  - `HermitianIndefinite` — \f$ WL \f$ Hermitian but \f$ W \f$ signed or
///    \f$ WL \f$ indefinite (the real signed-weight d'Alembertian on real
///    \f$ \ell^2 \f$). The interior equation is a stationarity condition.
///  - `NonNormal` — everything else (complex weights, complex
///    \f$ \ell^2 \f$). Certified block elimination with the left-kernel
///    compatibility check; no variational claim.
///
/// The spacetime path pairs the operator and its metric from one metric
/// source (`overCells`). Under the default `WhitneyPencil` the level is the
/// pencil \f$ (\tilde A_k^U, M_k^U) \f$ of `HodgeLaplacian::pencil`, regime
/// `ComplexSymmetricPencil` when both matrices are complex symmetric (a
/// trivial connection) and `NonNormal` otherwise; under `DiagonalWeights` it takes `HodgeLaplacian::laplacian(degree)` with
/// metric `HodgeLaplacian::weights(degree)` (the identity at degree zero) and
/// the regimes below.
///
/// ## Partitions
///
/// Components come from the discovered `PersistentModularity` partition
/// (vertex supports over the one-skeleton; a \f$ k \f$-cell belongs to a
/// component when all its vertices lie in the support) or from an explicit
/// caller-supplied cell partition, and may overlap. A cell is interior to
/// component \f$ v \f$ exactly when it is claimed only by \f$ v \f$ and every
/// nonzero coupling row and column of the operator stays inside \f$ v \f$'s
/// cells; every other cell is interface. Membership is matched by vertex set,
/// so a global relabeling yields an isomorphic reduction.
///
/// ## Interior nullspaces
///
/// On a spacetime-backed instance the topological interior zero modes are the
/// exact integer kernel of the stacked matrix
/// \f$ [\partial_k[:,I_v];\ \partial_{k+1}[I_v,:]^{\top}] \f$ (fraction-free
/// elimination; overflow fails loudly). The numerical kernel of the weighted
/// block, which gates solvability and the pseudoinverse, comes from
/// rank-revealing factorization and is cross-checked against the integer count
/// where both apply.
///
/// ## Caching and nesting
///
/// Per-component static contributions are cached in the shared
/// `AnalyticCache` keyed by the component's cell vertex-id set, so an accepted
/// local move (published as a `TouchedStar`) invalidates only the touched
/// component and its ancestry. `nextLevel` reduces the reduced operator again,
/// carrying parent/child lineage per coordinate; nested reduction equals
/// one-shot reduction whenever the elimination order is valid (the Schur
/// quotient property). Shifted factorizations are memoized per spectral
/// parameter within an instance.
///
/// This class is a read-only reduction of an already-relaxed operator.
class RecursiveQuotient {
  public:
    /// Reduction options. All tolerances are relative (scale-free).
    struct Options {
      Options();  // out-of-line so Options() can be an in-class default arg

      /// Certificate tolerance for `holds()` on the produced certificates.
      double tolerance{1e-10};
      /// Relative rank-revealing threshold for kernel/rank decisions.
      double rankTolerance{1e-9};
      /// Dimension at and above which dense kernels refuse. Per-component
      /// interior blocks below it may use dense rank-revealing (complete
      /// orthogonal) solves; at or above it only the sparse paths run.
      int denseCrossover{512};
      /// The declared labeled-sum Gram treatment for this run.
      FiberEmbeddingPolicy embeddingPolicy{FiberEmbeddingPolicy::CarryGramExactly};
      /// \f$ \varepsilon \f$ for `CertifiedNearIsometry`.
      double nearIsometryEpsilon{1e-10};
      /// Interior cells to retain as explicit stalk coordinates instead of
      /// eliminating (matrix path: fine indices).
      std::vector<int> selectedInteriorIndices{};
      /// The same for the spacetime path, as vertex-id tuples (matched by
      /// vertex set).
      std::vector<std::vector<std::uint64_t>> selectedInteriorCells{};
    };

    /// How this level was produced from its parent. The declared window and
    /// the producing step's residuals travel with the child.
    struct LevelProvenanceRead {
      /// The response step that produced this level.
      LevelOrigin origin{LevelOrigin::Base};
      /// The spectral parameter the parent pencil was evaluated at
      /// (`BandPencil` only; NaN otherwise, never 0).
      std::complex<double> lambda{std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()};
      /// Lower edge of the declared band/frequency window (`BandPencil`,
      /// `Surrogate`; NaN on `Base`/`StaticResponse`).
      double windowLower{std::numeric_limits<double>::quiet_NaN()};
      /// Upper edge of that window; NaN under the same conditions.
      double windowUpper{std::numeric_limits<double>::quiet_NaN()};
      /// Max relative interior solve residual of the producing step.
      double solveResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Max compatibility (left-kernel) violation of the producing step.
      double compatibilityResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Worst fine-space eigenresidual of the retained window pairs
      /// (`Surrogate`; NaN otherwise).
      double surrogateResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Smallest discarded fixed-interface eigenvalue minus the window upper
      /// edge (`Surrogate`; NaN otherwise).
      double discardedModeGap{std::numeric_limits<double>::quiet_NaN()};
      /// Whether the parent \f$ \lambda \f$ resonated with the interior
      /// spectrum, so the rank-deficient shifted block's kernel was retained.
      bool resonant{false};
      /// The producing step's own certificate, carried verbatim; a
      /// `Surrogate` level carries a certified-approximation certificate.
      Certificate certificate{};
    };

    /// One certified isolated band handed to `certifiedFiberSum` as a summand
    /// \f$ E_v \f$. The caller maps a band onto its component's fine
    /// coordinates (cells matched by vertex set, never by index).
    struct CertifiedBand {
      /// Owning component.
      int component{0};
      /// The band's right frame over this level's fine coordinates, flat
      /// row-major (`dimension()` x `rank`); its columns span the band.
      std::vector<std::complex<double>> frame{};
      /// The band's local left Riesz frame \f$ \tilde\Phi \f$ over the same
      /// coordinates, flat row-major (`dimension()` x `rank`), paired with
      /// `frame` by the plain transpose (\f$ \tilde\Phi^T \Phi = I \f$ on
      /// the band): `SpectralFiber::dualFrame()`, or a chain-level Riesz
      /// band's `leftFrame` carried into this level's coordinates. It is
      /// the left embedding, fixed before the overlap test and never
      /// replaced by a global dual. Empty: the level's metric dual stands in
      /// (see `certifiedFiberSum`). Either every band of a sum supplies one
      /// or none does.
      std::vector<std::complex<double>> leftFrame{};
      /// Band rank \f$ r_v \f$: the number of eigenvalues in the band.
      std::size_t rank{0};
      /// Distance to the nearest eigenvalue below the band. NaN when
      /// unknown.
      double lowerGap{std::numeric_limits<double>::quiet_NaN()};
      /// Distance to the nearest eigenvalue above the band; NaN when unknown.
      double upperGap{std::numeric_limits<double>::quiet_NaN()};
      /// The band's real extent [min Re, max Re]: the projection of its
      /// eigenvalues onto the real axis.
      double frequencyLower{std::numeric_limits<double>::quiet_NaN()};
      double frequencyUpper{std::numeric_limits<double>::quiet_NaN()};
      /// The band's window disc: the closed disc \f$ |z - c| \le \rho \f$
      /// in the complex spectral plane, with centre \f$ c \f$ and radius
      /// \f$ \rho \f$, that the band was certified on. A band of a non-normal
      /// or complex symmetric operator has a complex spectrum, and a disc is
      /// the region that encloses it; the real extent alone does not. Quiet
      /// NaN when the band was declared by its real extent only, in which case
      /// `certifiedFiberSum` derives the disc whose diameter is that extent.
      std::complex<double> windowCentre{std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN()};
      double windowRadius{std::numeric_limits<double>::quiet_NaN()};
      /// Whether the band met its producing configuration's certification
      /// thresholds. An uncertified band is still summed, and makes the
      /// labeled sum's certificate fail to hold.
      bool accepted{false};
      /// The band's certificate, carried verbatim onto the summand.
      Certificate certificate{};
    };

    /// The certificate data of one summand of a certified labeled sum.
    struct CertifiedFiberSummand {
      /// Owning component.
      int component{0};
      /// Nominal rank of the summand.
      std::size_t rank{0};
      /// Isolation gap below the band, from the producing band.
      double lowerGap{std::numeric_limits<double>::quiet_NaN()};
      /// Isolation gap above the band, from the producing band.
      double upperGap{std::numeric_limits<double>::quiet_NaN()};
      /// The band's real extent [min Re, max Re].
      double frequencyLower{std::numeric_limits<double>::quiet_NaN()};
      double frequencyUpper{std::numeric_limits<double>::quiet_NaN()};
      /// The band's window disc, centre and radius (see `CertifiedBand`):
      /// the band's own when it declared one, else the disc whose diameter is
      /// its real extent.
      std::complex<double> windowCentre{std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN()};
      double windowRadius{std::numeric_limits<double>::quiet_NaN()};
      /// Whether the producing band was accepted.
      bool accepted{false};
      /// The producing band's certificate.
      Certificate certificate{};
    };

    /// One retained stalk coordinate of the reduced space.
    struct RetainedCoordinate {
      /// Why this coordinate was retained.
      RetainedCoordinateKind kind{RetainedCoordinateKind::Interface};
      /// Owning component. A shared interface cell reports its first claiming
      /// component; all claimants are in `LabeledFiberSumRead`.
      int component{0};
      /// Fine-space index for `Interface`/`Selected` coordinates; -1 for
      /// mode coordinates (`Harmonic`/`Resonant`).
      int fineIndex{-1};
      /// The fine-space column vector this coordinate embeds to (length = fine
      /// dimension): an indicator for cell coordinates, the kernel-mode vector
      /// for mode coordinates.
      std::vector<std::complex<double>> embedding{};
      /// Human-readable provenance, e.g. "cell(3,7)", "harmonic[c1#0]",
      /// "resonant[c0#1@(2.5,0)]"; nested levels prefix "L<level>:".
      std::string provenance{};
    };

    /// Interior nullspace of one component (topological + numerical).
    struct InteriorNullspaceRead {
      /// The component described.
      int component{0};
      /// dim ker of the weighted interior block, at `rankTolerance`.
      std::size_t nullity{0};
      /// Exact integer topological zero-mode count (spacetime path): the
      /// combinatorial kernel of the stacked boundary blocks. Equals
      /// `integerBasis.size()`; 0 on the matrix path, so check
      /// `integerNullityMeasured` before comparing.
      std::size_t integerNullity{0};
      /// Whether the exact integer nullity was computed. False on the matrix
      /// path and on integer-kernel overflow, where `integerNullity == 0`
      /// means "not measured".
      bool integerNullityMeasured{false};
      /// `nullity - integerNullity`: nonzero when the numerical kernel is not
      /// the combinatorial one, a signed or complex metric having opened or
      /// closed a zero mode. NaN, not 0, when `integerNullityMeasured` is
      /// false.
      double nullityDiscrepancy{std::numeric_limits<double>::quiet_NaN()};
      /// Exact integer basis vectors over the component's interior cells, each
      /// of length `interiorIndices(component).size()` (spacetime path).
      std::vector<std::vector<long>> integerBasis{};
      /// Numerical right-kernel basis, flat row-major (|I_v| x nullity).
      std::vector<std::complex<double>> kernelBasis{};
      /// Numerical left-kernel basis of \f$ L_{II}^\dagger \f$, flat row-major
      /// (|I_v| x leftNullity). Equals the right kernel in the Hermitian
      /// regimes.
      std::vector<std::complex<double>> leftKernelBasis{};
      /// Measured \f$ \|L_{II} Z\| / \|L_{II}\| \f$ over the returned basis.
      Certificate certificate{};
    };

    /// The static reduction: the effective operator over interface plus
    /// retained coordinates, with per-coordinate provenance.
    struct StaticReductionRead {
      /// Fine indices of the kept cells (interface + selected), ascending. The
      /// reduced-coordinate order is kept cells ascending, then retained mode
      /// coordinates in component order.
      std::vector<int> interfaceIndices{};
      /// All reduced coordinates in order (size = reduced dimension).
      std::vector<RetainedCoordinate> coordinates{};
      /// The reduced operator, flat row-major (reducedDim x reducedDim); its
      /// leading interface block is \f$ L_{BB} - L_{BI} L_{II}^{+} L_{IB} \f$.
      std::vector<std::complex<double>> effectiveOperator{};
      /// Max relative interior solve residual
      /// \f$ \|L_{II}X - L_{IB}\| / \|L_{IB}\| \f$ across components.
      double solveResidual{0.0};
      /// Max compatibility violation \f$ \|Y^\dagger L_{IB}\| / \|L_{IB}\| \f$
      /// over interior (left-)kernels; 0 when every load is compatible.
      double compatibilityResidual{0.0};
      /// Static-domain certificate in the detected regime.
      Certificate certificate{};
    };

    /// One evaluation of the exact Feshbach--Schur response pencil.
    struct FeshbachRead {
      /// The spectral parameter the pencil was evaluated at.
      std::complex<double> lambda{};
      /// Declared band window, lower edge.
      double windowLower{0.0};
      /// Declared band window, upper edge.
      double windowUpper{0.0};
      /// \f$ F_B(\lambda) \f$ over the kept cells plus any resonant-retained
      /// modes, flat row-major.
      std::vector<std::complex<double>> response{};
      /// The coordinates of `response`: kept cells, then retained resonant
      /// modes.
      std::vector<RetainedCoordinate> coordinates{};
      /// Whether \f$ \lambda \f$ resonates with the interior spectrum, so a
      /// rank-deficient shifted block's kernel was retained.
      bool resonant{false};
      /// Max relative shifted solve residual across components.
      double solveResidual{0.0};
      /// Max resonant compatibility violation (left-kernel test); 0 when not
      /// resonant or compatible.
      double compatibilityResidual{0.0};
      /// Scale-normalized determinant-factorization residual
      /// \f$ |\det(L-\lambda) - \det(L_{II}-\lambda)\det F_B(\lambda)| \f$,
      /// measured below the dense crossover. NaN above it, at a resonance, and
      /// when the elimination was not certified.
      double determinantResidual{0.0};
      /// Band-window certificate in the detected regime.
      Certificate certificate{};
    };

    /// Multiplicity report at a candidate eigenvalue (band domain).
    struct MultiplicityRead {
      /// The candidate eigenvalue the contour is centred on.
      std::complex<double> lambda{};
      /// Radius of the counting contour.
      double contourRadius{0.0};
      /// Node count of the stabilized (doubled) evaluation.
      int nodes{0};
      /// Winding of \f$ \det F_B \f$ around the contour: zeros minus poles of
      /// the pencil determinant inside.
      int responseWinding{0};
      /// Winding of \f$ \det(L_{II} - z) \f$ around the contour: the
      /// interior-spectrum contribution, reported separately.
      int interiorWinding{0};
      /// Algebraic multiplicity of \f$ \operatorname{spec} L \f$ inside the
      /// contour: `responseWinding + interiorWinding`.
      int algebraic{0};
      /// \f$ \dim\ker F_B(\lambda) \f$ at `rankTolerance`.
      int geometric{0};
      /// Whether algebraic == geometric; guaranteed only in the self-adjoint
      /// or semisimple setting.
      bool semisimple{false};
      /// Max per-step phase advance / pi over both unwrapped determinant
      /// phases; an alias-free winding needs it well below 1.
      double phaseStepMargin{0.0};
      /// Certified-numerical winding certificate (stability + margin).
      Certificate certificate{};
    };

    /// Craig--Bampton / AMLS retained-mode surrogate over a declared window.
    struct CraigBamptonRead {
      /// The declared window disc: the closed disc \f$ |\theta - c| \le \rho \f$
      /// in the complex spectral plane with centre \f$ c \f$ and radius
      /// \f$ \rho \f$. A reduced eigenvalue is claimed exactly when it lies in
      /// it.
      std::complex<double> windowCentre{0.0, 0.0};
      double windowRadius{0.0};
      /// The retention radius \f$ \rho_{\text{cut}} \ge \rho \f$: a
      /// fixed-interface mode of a component is retained exactly when its
      /// eigenvalue \f$ \theta \f$ has \f$ |\theta - c| \le \rho_{\text{cut}} \f$,
      /// its distance from the centre in the complex plane.
      double retentionRadius{0.0};
      /// The real extent of the window disc, \f$ [\mathrm{Re}\,c - \rho,\
      /// \mathrm{Re}\,c + \rho] \f$: its projection onto the real axis, which
      /// is the window itself when the spectrum is real.
      double windowLower{0.0};
      double windowUpper{0.0};
      /// \f$ \mathrm{Re}\,c + \rho_{\text{cut}} \f$, the real extent of the
      /// retention disc's upper edge.
      double modeCutoff{0.0};
      /// Retained fixed-interface mode count per component.
      std::vector<int> retainedModes{};
      /// Reduction basis V, flat row-major (fineDim x reducedDim): interface
      /// unit block plus constraint modes, then fixed-interface modes.
      std::vector<std::complex<double>> basis{};
      /// Reduced stiffness \f$ V^\dagger W L V \f$, flat row-major
      /// (\f$ V^\dagger L V \f$ under the identity metric).
      std::vector<std::complex<double>> reducedStiffness{};
      /// Reduced mass \f$ V^\dagger W V \f$, flat row-major; Hermitian
      /// positive definite. The reduced eigenproblem is
      /// \f$ K y = \lambda M y \f$.
      std::vector<std::complex<double>> reducedMass{};
      /// \f$ \min_{\text{discarded}} |\theta - c| - \rho \f$: how far the
      /// nearest discarded fixed-interface eigenvalue sits outside the window
      /// disc; +inf when nothing was discarded. For a real spectrum and a
      /// discarded mode above the window this is the eigenvalue minus
      /// `windowUpper`.
      double discardedModeGap{0.0};
      /// The reduced eigenvalues inside the window disc, ascending by
      /// \f$ (\mathrm{Re}, \mathrm{Im}) \f$, and their real parts in the same
      /// order (the eigenvalues themselves when the spectrum is real).
      std::vector<std::complex<double>> windowSpectrum{};
      std::vector<double> windowEigenvalues{};
      /// Fine-space relative eigenresiduals
      /// \f$ \|L V y - \lambda V y\| / (\|L\|\,\|V y\|) \f$, one per window
      /// eigenvalue.
      std::vector<double> eigenResiduals{};
      /// Certified-approximation certificate against the declared residual
      /// tolerance.
      Certificate certificate{};
    };

    /// The abstract labeled sum \f$ \boxplus_v E_v \f$ with embedding and
    /// Gram data.
    struct LabeledFiberSumRead {
      /// Component index of each summand block, in embedding column order.
      std::vector<int> summandComponents{};
      /// Nominal rank \f$ r_v \f$ of each summand.
      std::vector<int> summandRanks{};
      /// The embedding \f$ J \f$ (the right embedding \f$ Y \f$) into the
      /// fine chain space, flat row-major (fineDim x totalRank). Columns are
      /// |W|-unit-normalized when the metric dual is the left embedding, and
      /// taken as given when an explicit left embedding is carried.
      std::vector<std::complex<double>> embedding{};
      /// The explicit left embedding \f$ \tilde Y \f$, flat row-major
      /// (fineDim x totalRank), assembled from the summands' local left Riesz
      /// frames and fixed before the overlap test. Empty when the level's
      /// metric dual is the left embedding.
      std::vector<std::complex<double>> leftEmbedding{};
      /// The overlap matrix, flat row-major (totalRank x totalRank):
      /// \f$ G = \tilde Y^T Y \f$ with an explicit left embedding;
      /// otherwise the metric pairing, \f$ G = J^\dagger W J \f$ on an
      /// operator level and \f$ G = J^T M J \f$ on a pencil level.
      std::vector<std::complex<double>> gram{};
      /// The declared policy this run proceeds by.
      FiberEmbeddingPolicy policy{FiberEmbeddingPolicy::CarryGramExactly};
      /// \f$ \|G - I\|_2 \f$.
      double gramDefect{0.0};
      /// \f$ \dim\ker G \f$ at `rankTolerance`: the labeled-sum overcounting,
      /// 0 exactly when the internal sum is direct.
      std::size_t quotientNullity{0};
      /// Total rank of the labeled sum: \f$ \sum_v r_v \f$ nominal.
      std::size_t nominalRank{0};
      /// Effective rank after the declared treatment:
      /// nominal for `CarryGramExactly`/`CertifiedNearIsometry`,
      /// \f$ \operatorname{rank} G \f$ for `QuotientKernel`.
      std::size_t effectiveRank{0};
      /// Orthonormal basis of \f$ (\ker G)^\perp \f$, the complement of the
      /// right radical, flat row-major (totalRank x effectiveRank); populated
      /// under `QuotientKernel`.
      std::vector<std::complex<double>> quotientBasis{};
      /// The left partner \f$ L_q \f$ of `quotientBasis` \f$ R_q \f$, same
      /// shape, so that the quotient of any one-particle matrix is
      /// \f$ L_q^T X R_q \f$ — one transpose pairing: \f$ \bar R_q \f$ for a
      /// Hermitian (metric) Gram, and \f$ \bar U_r \f$ from the Gram's SVD
      /// (the complement of the left radical) for a bilinear one, so that
      /// \f$ L_q^T G R_q \f$ is nondegenerate. Populated under
      /// `QuotientKernel`.
      std::vector<std::complex<double>> leftQuotientBasis{};
      /// Whether the summands are certified isolated bands \f$ E_v \f$: false
      /// for `labeledFiberSum()`, true for `certifiedFiberSum()`.
      bool fromCertifiedBands{false};
      /// Per-summand band certificates, populated only when
      /// `fromCertifiedBands`; empty otherwise.
      std::vector<CertifiedFiberSummand> summandCertificates{};
      /// The smallest isolation gap over the summed bands. NaN when not summed
      /// from certified bands or when every gap is unknown.
      double worstIsolationGap{std::numeric_limits<double>::quiet_NaN()};
      /// Whether every summed band was accepted by its producing
      /// configuration. False, with the certificate failing to hold, when any
      /// summand is uncertified.
      bool allBandsAccepted{false};
      /// Certificate of the declared policy's claim.
      Certificate certificate{};
    };

    /// The Fock stage \f$ \HK_{\ell+1} = \Fock(\hh_{\ell+1}) \f$ over the
    /// labeled sum, carried at the spectrum level. The \f$ 2^M \f$ vector is
    /// never allocated; the read refuses past the declared term budget.
    struct FockStageRead {
      /// \f$ M = \dim\hh_{\ell+1} \f$: the labeled sum's effective rank.
      std::size_t modes{0};
      /// The policy the labeled sum was treated by.
      FiberEmbeddingPolicy policy{FiberEmbeddingPolicy::CarryGramExactly};
      /// \f$ \|G - I\| \f$ of the underlying labeled sum, carried through.
      double gramDefect{std::numeric_limits<double>::quiet_NaN()};
      /// The one-particle operator on the labeled-sum basis, compressed in
      /// the same pairing as `gram`: \f$ h = \tilde Y^T \mathcal L Y \f$
      /// with an explicit left embedding (\f$ \mathcal L = L \f$ on an
      /// operator level, \f$ M^{-1}\tilde A \f$ on a pencil level);
      /// otherwise \f$ h = J^\dagger W L J \f$ on an operator level and
      /// \f$ h = J^T \tilde A J \f$ on a pencil level. Restricted to the
      /// quotient \f$ L_q^T h R_q \f$ under `QuotientKernel`; flat row-major
      /// (modes x modes).
      std::vector<std::complex<double>> oneParticle{};
      /// The Gram \f$ G \f$ on the same basis, so a `CarryGramExactly` run
      /// can pair \f$ h \f$ against it rather than assume orthonormality.
      std::vector<std::complex<double>> gram{};
      /// The pairing both were compressed in: "metric-hermitian" (operator
      /// level, \f$ J^\dagger W \f$), "metric-transpose" (pencil level,
      /// \f$ J^T \f$ against \f$ M \f$), or "left-embedding" (explicit
      /// \f$ \tilde Y^T \f$).
      std::string pairing{};
      /// Eigenvalues of \f$ h \f$, ascending by (Re, Im).
      std::vector<std::complex<double>> oneParticleSpectrum{};
      /// \f$ \dim\Fock(\hh) = 2^M \f$ as a double; exact through 2^53, +inf
      /// beyond.
      double fockDimension{std::numeric_limits<double>::quiet_NaN()};
      /// Whether the free many-body spectrum below was materialized.
      bool spectrumMaterialized{false};
      /// The exact free many-body spectrum of \f$ d\Gamma(h) \f$: all
      /// \f$ 2^M \f$ occupation subset sums, ascending. Empty when the budget
      /// refused it.
      std::vector<std::complex<double>> fockSpectrum{};
      /// Certificate of the one-particle compression.
      Certificate certificate{};
    };

    /// One operator-valued link of the next-level response network.
    struct ResponseEdge {
      /// Source component of the link.
      int from{0};
      /// Target component of the link.
      int to{0};
      /// The effective block between the stalks, flat row-major
      /// (stalkDim(from) x stalkDim(to)).
      std::vector<std::complex<double>> block{};
    };

    /// The next-level operator-valued response network.
    struct ResponseNetworkRead {
      /// Stalk dimension per component: interface cells claimed plus retained
      /// interior modes owned.
      std::vector<int> stalkDimensions{};
      /// Reduced-coordinate indices of each stalk. A shared interface cell
      /// appears in every claiming stalk, so the network asserts no internal
      /// direct sum.
      std::vector<std::vector<int>> stalkCoordinates{};
      /// Diagonal blocks (one per component), flat row-major.
      std::vector<std::vector<std::complex<double>>> vertexBlocks{};
      /// Off-diagonal links (only nonzero or stalk-sharing pairs).
      std::vector<ResponseEdge> edges{};
      /// Largest |entry| of the reduced operator not covered by any
      /// vertex/edge block; 0 means the network reproduces the operator.
      double coverageResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Exact-tiling certificate (residual = uncovered magnitude).
      Certificate certificate{};
    };

    /// A cellular-sheaf (or simplicial) realization of the response network,
    /// emitted only when the restriction maps reproduce the blocks.
    struct SheafRealizationRead {
      /// Whether a certified realization was emitted. When false the maps
      /// below are empty and the general response network is retained.
      bool emitted{false};
      /// Whether every stalk is one-dimensional: a weighted simplicial
      /// 1-complex realization.
      bool simplicial{false};
      /// Edge stalk dimension per network edge.
      std::vector<int> edgeStalkDimensions{};
      /// Restriction maps per network edge: for e = (u, v),
      /// \f$ \rho_{u\to e} \f$ (edgeDim x stalkDim(u)) then
      /// \f$ \rho_{v\to e} \f$ (edgeDim x stalkDim(v)), flat row-major.
      std::vector<std::vector<std::complex<double>>> restrictionMaps{};
      /// Max relative reconstruction residual of the sheaf Laplacian against
      /// the response network blocks.
      double reconstructionResidual{std::numeric_limits<double>::quiet_NaN()};
      /// Realization certificate; `holds()` gates `emitted`.
      Certificate certificate{};
    };

    /// Build over an explicit operator (fixtures and next-level recursion).
    /// `op` is flat row-major (`dim` x `dim`); `weights` is the diagonal
    /// chain-space metric \f$ W \f$ (empty = identity); `components` are
    /// 0-based fine index sets, possibly overlapping, whose union must cover
    /// every index.
    /// @throws std::invalid_argument on malformed sizes/partition.
    [[nodiscard]] static RecursiveQuotient overMatrix(
        const std::vector<std::complex<double>> &op, int dim,
        const std::vector<std::complex<double>> &weights,
        const std::vector<std::vector<int>> &components,
        const Options &options = Options());
    /// Build over a symmetric pencil \f$ (\tilde A, M) \f$ on geometric
    /// images (the chain-level Whitney Hodge pencil). `A` and `M` are flat
    /// row-major `dim` x `dim`, with `M` the sparse complex-symmetric inverse
    /// chain metric (base level) or the carried Gram \f$ \mathcal G \f$
    /// (child level). Every shifted elimination is taken on
    /// \f$ \mathcal P(\lambda) = \tilde A - \lambda M \f$, and the static
    /// reduction at \f$ \lambda = 0 \f$ coincides with the operator path. A
    /// child carries \f$ \mathcal G_{\ell+1} = T^T M T \f$ with \f$ T \f$ the
    /// constraint modes; labeled-sum Grams on a pencil level are
    /// \f$ J^T M J \f$. The Hermitian surrogate's \f$ M^{-1/2} \f$
    /// orthonormalization is not applied here; `nextLevelFromSurrogate`
    /// carries the congruence instead.
    [[nodiscard]] static RecursiveQuotient overPencil(
        const std::vector<std::complex<double>> &A,
        const std::vector<std::complex<double>> &M, int dim,
        const std::vector<std::vector<int>> &components,
        const Options &options = Options());
    /// Whether this level is a pencil level (see `overPencil`).
    [[nodiscard]] bool isPencil() const noexcept { return pencil_; }
    /// The pencil's metric \f$ M \f$ (base) or carried Gram \f$ \mathcal G \f$
    /// (child), flat row-major `dim` x `dim`; empty on an operator level.
    [[nodiscard]] std::vector<std::complex<double>> pencilMetric() const;

    /// Build over a spacetime's Hodge operator at `degree`, components given
    /// as explicit k-cell sets (each cell a vertex-id tuple, matched by vertex
    /// set). An `AnalyticCache` bound to the same spacetime enables
    /// per-component reuse across accepted moves.
    ///
    /// The operator and its metric come from one source, `metricSource`,
    /// which defaults to the process-wide
    /// `HodgeLaplacian::defaultMetricSource()`. Under `WhitneyPencil` (the
    /// default) the level is a pencil level over the dressed Whitney pencil
    /// $ (	ilde A_k^U, M_k^U) $ of `HodgeLaplacian::pencil`, exactly as
    /// `overPencil` builds one, so the reduction moves with the connection
    /// $ U $. Under `DiagonalWeights` it is an operator level over
    /// `HodgeLaplacian::laplacian(degree)` with the diagonal metric
    /// `HodgeLaplacian::weights(degree)`.
    /// @throws std::invalid_argument on an unknown cell or uncovered cells.
    [[nodiscard]] static RecursiveQuotient overCells(
        std::shared_ptr<Spacetime> st, int degree,
        const std::vector<std::vector<std::vector<std::uint64_t>>> &componentCells,
        const Options &options = Options(),
        std::shared_ptr<AnalyticCache> cache = nullptr,
        HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());

    /// Build over a spacetime's Hodge operator at `degree`, components given
    /// as vertex supports: a k-cell belongs to a component when all its
    /// vertices lie in the support. Cells claimed by no support are gathered
    /// into one residual component appended after the supplied ones. The
    /// operator and metric are those of `overCells` under `metricSource`.
    [[nodiscard]] static RecursiveQuotient overVertexSupports(
        std::shared_ptr<Spacetime> st, int degree,
        const std::vector<std::vector<std::uint64_t>> &componentVertexSupports,
        const Options &options = Options(),
        std::shared_ptr<AnalyticCache> cache = nullptr,
        HodgeLaplacian::MetricSource metricSource = HodgeLaplacian::defaultMetricSource());

    /// The metric source of a spacetime-backed level (see `overCells`);
    /// `std::nullopt` on the matrix and pencil paths and on child levels.
    [[nodiscard]] std::optional<HodgeLaplacian::MetricSource> metricSource() const noexcept {
      return metricSource_;
    }

    /// Fine dimension (number of k-cells / coordinates at this level).
    [[nodiscard]] int dimension() const noexcept { return dim_; }
    /// Number of components.
    [[nodiscard]] int componentCount() const noexcept {
      return static_cast<int>(components_.size());
    }
    /// Hodge degree (spacetime paths; -1 on the matrix path).
    [[nodiscard]] int degree() const noexcept { return degree_; }
    /// Nesting level: 0 for a base instance, parent + 1 under `nextLevel`.
    [[nodiscard]] int level() const noexcept { return level_; }
    /// The regime detected for the operator (see `CertificateRegime`).
    [[nodiscard]] CertificateRegime regime() const noexcept { return regime_; }

    /// Ascending fine indices of the kept cell coordinates: the interface
    /// cells \f$ B \f$ plus any caller-selected retained interior cells.
    /// `StaticReductionRead::coordinates` distinguishes the kinds.
    [[nodiscard]] const std::vector<int> &interfaceIndices() const noexcept {
      return interfaceIndices_;
    }
    /// Fine indices of component `component`'s interior cells, ascending.
    /// @throws std::out_of_range on a bad component index.
    [[nodiscard]] const std::vector<int> &interiorIndices(int component) const;

    /// The k-cell vertex tuples of this level's fine coordinates, in
    /// coordinate order. Spacetime paths only: empty on the matrix path and on
    /// child levels. A `CertifiedBand`'s cells are matched against these by
    /// vertex set, never by index.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &cellVertices()
        const noexcept {
      return cellVertices_;
    }

    /// Provenance of each fine coordinate: cell vertex tuples on the spacetime
    /// path, inherited reduced-coordinate provenance under `nextLevel`.
    [[nodiscard]] const std::vector<std::string> &coordinateProvenance()
        const noexcept {
      return provenance_;
    }

    /// The interior nullspace of one component: the integer topological basis
    /// on the spacetime path, the numerical kernel and left kernel always.
    /// @throws std::out_of_range on a bad component index.
    [[nodiscard]] InteriorNullspaceRead interiorNullspace(int component) const;

    /// The exact supported static reduction, memoized; per-component
    /// contributions are served from the bound `AnalyticCache` when fresh.
    [[nodiscard]] const StaticReductionRead &staticReduction() const;

    /// Verify the regime-appropriate static certificate on one kept-cell
    /// probe `b` (length = `interfaceIndices().size()`): minimized fine energy
    /// \f$ x^\dagger WLx \f$ vs \f$ b^\dagger (WL_{\text{eff}}) b \f$ in the
    /// positive regime, interior stationarity in the Hermitian-indefinite one,
    /// block elimination plus the left-kernel check in the non-normal one.
    /// Retained mode coordinates are held at zero.
    /// @throws std::invalid_argument on size mismatch.
    [[nodiscard]] Certificate staticProbeCertificate(
        const std::vector<std::complex<double>> &probe) const;

    /// The worst `staticProbeCertificate` over the deterministic probe set:
    /// every interface basis vector and the all-ones vector.
    [[nodiscard]] Certificate verifyStatic() const;

    /// Evaluate the exact Feshbach--Schur response \f$ F_B(\lambda) \f$ over
    /// a caller-supplied window (plain lower/upper frequencies). Shifted
    /// factorizations are memoized per \f$ \lambda \f$.
    /// @throws std::invalid_argument when `windowLower > windowUpper`.
    [[nodiscard]] FeshbachRead feshbach(std::complex<double> lambda,
                                        double windowLower,
                                        double windowUpper) const;

    /// Multiplicity report at `lambda`: algebraic from the winding of the
    /// unwrapped determinant phases of \f$ \det F_B(\cdot) \f$ and
    /// \f$ \det(L_{II} - \cdot) \f$ around the circle of `radius`, geometric
    /// from \f$ \dim\ker F_B(\lambda) \f$. The winding is validated by
    /// doubling the node count until stable.
    /// @throws std::invalid_argument on a non-positive radius or nodes < 8.
    [[nodiscard]] MultiplicityRead multiplicity(std::complex<double> lambda,
                                                double radius,
                                                int nodes = 64) const;

    /// Craig--Bampton retained-mode basis over a declared window disc. The
    /// window is the closed disc \f$ |\theta - c| \le \rho \f$ in the complex
    /// spectral plane with centre \p windowCentre and radius \p windowRadius;
    /// a component's fixed-interface mode is retained exactly when its
    /// eigenvalue lies within \p retentionRadius of the centre, and a reduced
    /// eigenvalue is claimed exactly when it lies in the window. A disc,
    /// because the spectrum of a non-normal or complex symmetric level is
    /// complex and an interval on the real axis does not enclose it; the rule
    /// is the same in every regime, and on a real spectrum it reads as the
    /// interval \f$ [\mathrm{Re}\,c - \rho, \mathrm{Re}\,c + \rho] \f$.
    /// `residualTolerance` is the declared acceptance residual the certificate
    /// holds against; negative selects the strict `Options::tolerance`, under
    /// which a truncated surrogate reports `holds() == false` while still
    /// carrying its window, gap and residuals.
    ///
    /// The surrogate exists in every regime; what the regime decides is the
    /// pairing. In `PositiveSemidefinite` and `HermitianIndefinite` the pairing
    /// is the adjoint against the positive diagonal chain metric \f$ W \f$, the
    /// fixed-interface modes come from a self-adjoint eigensolver on
    /// \f$ W^{1/2}LW^{-1/2} \f$, and the reduced pair
    /// \f$ (V^\dagger WLV,\ V^\dagger WV) \f$ is solved as a Hermitian
    /// generalized eigenproblem. In `NonNormal` and `ComplexSymmetricPencil`
    /// the pairing is the transpose against the level's own metric — the
    /// carried Gram \f$ \mathcal G \f$ on a pencil level, the diagonal weights
    /// otherwise — the fixed-interface modes come from a general complex
    /// eigensolver on the interior pencil \f$ (L_{II}, W_{II}) \f$, and the
    /// reduced pair \f$ (V^TWLV,\ V^TWV) \f$ is solved the same way. No adjoint
    /// is formed and no definiteness is assumed on that path.
    ///
    /// This is a certified approximation and never an exact spectral identity:
    /// the error is controlled by the residuals of the retained pairs and by
    /// `discardedModeGap`, both of which the read carries. The pencil-level
    /// counterpart that holds a surrogate spectrum to the exact Feshbach map is
    /// `chainhodge::PencilSchur::craigBampton`.
    /// @throws std::invalid_argument on an indefinite metric in a Hermitian
    ///   regime, a singular interior or reduced chain metric in a bilinear one,
    ///   a negative radius, or `retentionRadius < windowRadius` (a mode inside
    ///   the window is never discarded); std::length_error when a component's
    ///   interior block is at or above the dense crossover.
    [[nodiscard]] CraigBamptonRead craigBampton(
        std::complex<double> windowCentre, double windowRadius,
        double retentionRadius, double residualTolerance = -1.0) const;

    /// The same surrogate declared by a real window: the interval
    /// \f$ [a, b] \f$ = [\p windowLower, \p windowUpper] is the disc with
    /// centre \f$ (a+b)/2 \f$ and radius \f$ (b-a)/2 \f$, and \p modeCutoff
    /// is the retention disc's real upper edge, so the retention radius is
    /// \f$ \rho_{\text{cut}} = \text{modeCutoff} - (a+b)/2 \f$. On a real
    /// spectrum a mode is therefore retained exactly when
    /// \f$ a + b - \text{modeCutoff} \le \theta \le \text{modeCutoff} \f$,
    /// which is every mode at or below the cutoff whenever no eigenvalue lies
    /// below \f$ a + b - \text{modeCutoff} \f$ — the case of a positive
    /// operator with a window that starts at or below its lowest eigenvalue.
    /// @throws std::invalid_argument when `windowLower > windowUpper` or
    ///   `modeCutoff < windowUpper`, and as the disc form otherwise.
    [[nodiscard]] CraigBamptonRead craigBampton(
        double windowLower, double windowUpper, double modeCutoff,
        double residualTolerance = -1.0) const;

    /// The abstract labeled retained-fiber sum with embedding and Gram data,
    /// treated by the run's declared `FiberEmbeddingPolicy`. The summands are
    /// the reduction's retained coordinates — a component's claimed interface
    /// cells plus the interior modes it owns — and carry no band certificate.
    [[nodiscard]] LabeledFiberSumRead labeledFiberSum() const;

    /// The labeled sum \f$ \boxplus_v E_v \f$ over certified isolated bands,
    /// each with its isolation gap and certificate carried onto its summand.
    /// Bands are summed in the order given; an uncertified band is summed and
    /// reported rather than dropped, and makes the sum's certificate fail to
    /// hold.
    ///
    /// When the bands carry their local left Riesz frames
    /// (`CertifiedBand::leftFrame`), those assemble the left embedding
    /// \f$ \tilde Y \f$, fixed before the test, and the overlap certificate is
    /// \f$ G = \tilde Y^T Y \f$ with defect \f$ \Delta G = G - I \f$ — the
    /// exact complex amplitude error \f$ \tilde a^T \Delta G\, b \f$. Frames
    /// are then taken as given (no column normalization, which would break
    /// their pairing). Without left frames the level's metric dual stands in.
    /// @throws std::invalid_argument on a frame whose size is not
    ///   `dimension() * rank`, a band naming an unknown component, or a sum
    ///   in which some bands carry a left frame and others do not.
    [[nodiscard]] LabeledFiberSumRead certifiedFiberSum(
        const std::vector<CertifiedBand> &bands) const;

    /// The Fock stage \f$ \Fock(\boxplus_v E_v) \f$ over a labeled sum: the
    /// one-particle compression, its spectrum, and the free many-body
    /// spectrum of \f$ d\Gamma(h) \f$ as occupation subset sums. One pairing
    /// throughout: \f$ h \f$ is compressed in exactly the pairing the sum's
    /// Gram \f$ G \f$ was built in — \f$ \tilde Y^T \mathcal L Y \f$ against
    /// \f$ G = \tilde Y^T Y \f$ with an explicit left embedding,
    /// \f$ J^T \tilde A J \f$ against \f$ J^T M J \f$ on a pencil level, and
    /// \f$ J^\dagger W L J \f$ against \f$ J^\dagger W J \f$ on an operator
    /// level (the Hermitian special case) — and a quotient takes
    /// \f$ L_q^T (\cdot) R_q \f$ of both. `maxTerms` bounds the materialized
    /// many-body spectrum; beyond it the read refuses
    /// (`spectrumMaterialized == false`) rather than allocating \f$ 2^M \f$
    /// entries.
    /// @throws std::invalid_argument when the sum's embedding does not match
    ///   this level's dimension.
    [[nodiscard]] FockStageRead fockStage(
        const LabeledFiberSumRead &sum,
        std::size_t maxTerms = std::size_t{1} << 22) const;

    /// \f$ P = \mathrm{PersistentPartition}(\RN) \f$: the component-discovery
    /// step at any scale. Partitions the coordinates of an operator-valued
    /// response network by persistent modularity over its off-diagonal
    /// magnitude graph \f$ w_{ij} = |R_{ij}| + |R_{ji}| \f$; the diagonal
    /// never enters. Modularity only proposes candidate supports; it never
    /// vetoes a certified fiber. Isolated coordinates come back as singleton
    /// components, so the partition covers every index exactly once.
    ///
    /// Reference: Newman and Girvan, "Finding and evaluating community
    /// structure in networks", arXiv:cond-mat/0308217
    /// @throws std::invalid_argument on a malformed operator size or a
    ///   non-positive restart count.
    [[nodiscard]] static std::vector<std::vector<int>> persistentPartition(
        const std::vector<std::complex<double>> &op, int dim,
        double gamma = 1.0, int restarts = 4, std::uint64_t baseSeed = 0);

    /// `persistentPartition` over a declared window of resolutions rather than
    /// at one.
    ///
    /// The resolution parameter \f$ \gamma \f$ is a free knob of the proposer,
    /// and modularity is subject to the resolution limit, so a community that
    /// stands at one value of \f$ \gamma \f$ and nowhere else states nothing
    /// about the operator. This form scans `gammas` in the order given,
    /// follows each community across adjacent resolutions by support overlap,
    /// and keeps only the components whose persistence track covers the whole
    /// window, taking each track's member at the FIRST resolution of the
    /// window as the support it proposes. Every coordinate no such component
    /// claimed comes back as a singleton, so the result still covers every
    /// index exactly once and is still a partition to hand to `nextLevel`.
    ///
    /// A window of one resolution is the single-resolution form, since a track
    /// over one slice covers its window.
    /// @throws std::invalid_argument on a malformed operator size, an empty
    ///   window, or a non-positive restart count.
    [[nodiscard]] static std::vector<std::vector<int>> persistentPartition(
        const std::vector<std::complex<double>> &op, int dim,
        const std::vector<double> &gammas, int restarts = 4,
        std::uint64_t baseSeed = 0);

    /// `persistentPartition` of this level's reduced operator: the partition
    /// \f$ P_\ell \f$ to hand straight to `nextLevel`, as
    /// `child = parent.nextLevel(parent.childPersistentPartition())`.
    [[nodiscard]] std::vector<std::vector<int>> childPersistentPartition(
        double gamma = 1.0, int restarts = 4, std::uint64_t baseSeed = 0) const;

    /// `childPersistentPartition` over a declared window of resolutions: the
    /// components of this level's reduced operator that persist across the
    /// whole window (see the window form of `persistentPartition`).
    [[nodiscard]] std::vector<std::vector<int>> childPersistentPartition(
        const std::vector<double> &gammas, int restarts = 4,
        std::uint64_t baseSeed = 0) const;

    /// The composable amplitude budget of the `CertifiedNearIsometry`
    /// policy: two embeddings with Gram defects \f$ \varepsilon_A,
    /// \varepsilon_B \f$ compose (tensor) to at most
    /// \f$ \varepsilon_{AB} \le \varepsilon_A + \varepsilon_B +
    /// \varepsilon_A\varepsilon_B \f$, and the amplitude error obeys
    /// \f$ |a^\dagger G b - a^\dagger b| \le \varepsilon\|a\|\|b\| \f$.
    [[nodiscard]] static double composeNearIsometryBudget(
        double epsilonA, double epsilonB) noexcept {
      return epsilonA + epsilonB + epsilonA * epsilonB;
    }

    /// The next-level operator-valued response network: component stalks and
    /// the effective blocks of the static reduction.
    [[nodiscard]] ResponseNetworkRead responseNetwork() const;

    /// Attempt the cellular-sheaf / simplicial realization of the response
    /// network. When the blocks are not reproduced, `emitted == false` with the
    /// failing residual on the certificate. Hermitian regimes only.
    [[nodiscard]] SheafRealizationRead sheafRealization() const;

    /// Reduce again at \f$ \lambda = 0 \f$: a child quotient over this level's
    /// static reduced operator, with `components` indexing the reduced
    /// coordinates. The child inherits provenance ("L<level>:" prefixes),
    /// level + 1, and this level's chain metric restricted through the reduced
    /// coordinates.
    [[nodiscard]] RecursiveQuotient nextLevel(
        const std::vector<std::vector<int>> &components,
        const Options &options) const;

    /// `nextLevel` with this instance's options.
    [[nodiscard]] RecursiveQuotient nextLevel(
        const std::vector<std::vector<int>> &components) const;

    /// Reduce again on the pencil at a declared \f$ \lambda \f$ over a
    /// declared band window; the child's operator is \f$ F_B(\lambda) \f$
    /// rather than the static complement. `components` index the pencil's
    /// reduced coordinates, which include any resonant modes retained at
    /// \f$ \lambda \f$ and so need not match the static reduction's; use
    /// `persistentPartition(feshbach(...).response, ...)` to discover them.
    /// @throws std::invalid_argument when `windowLower > windowUpper` or the
    ///   partition does not cover the pencil's coordinates.
    [[nodiscard]] RecursiveQuotient nextLevelAtLambda(
        const std::vector<std::vector<int>> &components,
        std::complex<double> lambda, double windowLower, double windowUpper,
        const Options &options) const;

    /// `nextLevelAtLambda` with this instance's options.
    [[nodiscard]] RecursiveQuotient nextLevelAtLambda(
        const std::vector<std::vector<int>> &components,
        std::complex<double> lambda, double windowLower,
        double windowUpper) const;

    /// Reduce again through a certified linear surrogate: the cached
    /// Craig--Bampton/AMLS reduction over a declared frequency window, as a
    /// child level. The surrogate's reduced pencil
    /// \f$ (K, M) = (V^\dagger W L V, V^\dagger W V) \f$ is generalized while
    /// a level carries a diagonal chain metric, so the child is built on the
    /// \f$ M \f$-orthonormalized basis \f$ V M^{-1/2} \f$: operator
    /// \f$ M^{-1/2} K M^{-1/2} \f$, metric the identity. The congruence
    /// preserves the generalized eigenvalues of \f$ (K, M) \f$ exactly, so the
    /// approximation is entirely in the truncation, which the child's
    /// certificate and `discardedModeGap` report.
    /// @throws as `craigBampton`, plus std::invalid_argument when the
    ///   partition does not cover the surrogate's coordinates.
    [[nodiscard]] RecursiveQuotient nextLevelFromSurrogate(
        const std::vector<std::vector<int>> &components, double windowLower,
        double windowUpper, double modeCutoff, double residualTolerance,
        const Options &options) const;

    /// `nextLevelFromSurrogate` with this instance's options.
    [[nodiscard]] RecursiveQuotient nextLevelFromSurrogate(
        const std::vector<std::vector<int>> &components, double windowLower,
        double windowUpper, double modeCutoff,
        double residualTolerance = -1.0) const;

    /// How this level was produced from its parent, with the declared window
    /// and the producing step's residuals and certificate.
    [[nodiscard]] const LevelProvenanceRead &levelProvenance() const noexcept {
      return levelProvenance_;
    }

    /// Drop memoized reductions and factorizations and, on the spacetime path,
    /// re-read the operator values for the same cell complex; a structural move
    /// needs a fresh instance. The bound `AnalyticCache` still gates
    /// per-component reuse, so the next `staticReduction` recomputes only the
    /// invalidated components.
    void invalidate();

    /// The options this instance runs with.
    [[nodiscard]] const Options &options() const noexcept { return options_; }

  private:
    struct ComponentSolve;  // per-component factorization + kernel payload

    RecursiveQuotient() = default;

    void initMatrix(const std::vector<std::complex<double>> &op, int dim,
                    const std::vector<std::complex<double>> &weights,
                    const std::vector<std::vector<int>> &components,
                    const Options &options);
    void classify();
    // Measures the regime from the operator and its carried metric.
    void detectRegime();
    [[nodiscard]] std::shared_ptr<ComponentSolve> componentSolve(
        int component) const;
    [[nodiscard]] std::shared_ptr<ComponentSolve> computeSolve(
        int component, std::complex<double> lambda) const;
    [[nodiscard]] const std::vector<std::shared_ptr<ComponentSolve>> &
    shiftedSolves(std::complex<double> lambda) const;
    [[nodiscard]] std::vector<std::complex<double>> contourDeterminants(
        std::complex<double> lambda, double radius, int nodes,
        std::vector<std::complex<double>> &interiorDets) const;
    [[nodiscard]] static int windingFromPhases(
        const std::vector<std::complex<double>> &values, double *maxStep);
    [[nodiscard]] std::vector<std::uint64_t> componentVertexIds(
        int component) const;
    [[nodiscard]] std::vector<long> integerKernelStack(int component,
                                                       int *rows) const;
    // Build a child level over `op` (reduced x reduced) with the chain metric
    // induced by `coordinates`' embeddings through W.
    [[nodiscard]] RecursiveQuotient childOver(
        const std::vector<std::complex<double>> &op,
        const std::vector<RetainedCoordinate> &coordinates,
        const std::vector<std::vector<int>> &components,
        const Options &options) const;
    // Pencil child: the same reduced operator, with the carried Gram T^T M T
    // over the constraint modes of `solves` and the retained resonant
    // embeddings.
    [[nodiscard]] RecursiveQuotient pencilChildOver(
        const std::vector<std::complex<double>> &op,
        const std::vector<RetainedCoordinate> &coordinates,
        const std::vector<std::vector<int>> &components,
        const Options &options,
        const std::vector<std::shared_ptr<ComponentSolve>> &solves) const;
    [[nodiscard]] Eigen::MatrixXcd pencilConstraintModes(
        const std::vector<RetainedCoordinate> &coordinates,
        const std::vector<std::shared_ptr<ComponentSolve>> &solves) const;
    // The Gram/policy treatment shared by both labeled-sum entry points.
    // `leftColumns` is the explicit left embedding (empty: the metric dual).
    [[nodiscard]] LabeledFiberSumRead summarizeFiberSum(
        const std::vector<Eigen::VectorXcd> &columns,
        const std::vector<Eigen::VectorXcd> &leftColumns = {}) const;

    // --- problem data (op_/weights_ refresh under invalidate()) ------------
    Eigen::SparseMatrix<std::complex<double>> op_{};
    Eigen::VectorXcd weights_{};        // diagonal chain metric W
    bool pencil_{false};                // pencil level: shifts by lambda*M, Gram carried
    Eigen::SparseMatrix<std::complex<double>> pencilMetric_{};  // M (base) or G (child)
    double opNorm_{0.0};                // scale for relative residuals
    int dim_{0};
    int degree_{-1};
    int level_{0};
    LevelProvenanceRead levelProvenance_{};
    CertificateRegime regime_{CertificateRegime::NonNormal};
    Options options_{};
    std::vector<std::vector<int>> components_{};       // claimed cells
    std::vector<std::vector<int>> interior_{};         // per component
    std::vector<int> interfaceIndices_{};              // kept cells, ascending
    std::vector<RetainedCoordinateKind> keptKinds_{};  // Interface/Selected
    std::vector<int> keptOwner_{};                     // first claimant
    std::vector<int> interfacePosition_{};             // fine -> kept position
    std::vector<std::vector<int>> claimants_{};        // fine -> components
    std::vector<std::string> provenance_{};            // fine coordinates
    std::uint64_t partitionFingerprint_{0};            // cache-kind qualifier
    std::shared_ptr<Spacetime> st_{};
    std::shared_ptr<AnalyticCache> cache_{};
    // spacetime path: the one source of the operator and its metric
    std::optional<HodgeLaplacian::MetricSource> metricSource_{};
    // spacetime path extras: per-cell vertex tuples + integer boundary maps
    std::vector<std::vector<std::uint64_t>> cellVertices_{};
    bool hasBoundary_{false};
    std::vector<long> boundaryK_{};                    // ∂_degree, flat
    int boundaryKRows_{0};
    std::vector<long> boundaryK1_{};                   // ∂_{degree+1}, flat
    int boundaryK1Cols_{0};

    // --- memoized results ---------------------------------------------------
    mutable std::optional<StaticReductionRead> static_{};
    mutable std::vector<std::shared_ptr<ComponentSolve>> solves_{};
    mutable std::map<std::pair<double, double>,
                     std::vector<std::shared_ptr<ComponentSolve>>>
        shifted_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_RECURSIVEQUOTIENT_H
