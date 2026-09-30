// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_SELFCONSISTENTMEANFIELD_H
#define TESSERA_COBORDISM_SELFCONSISTENTMEANFIELD_H

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include "cobordism/HolomorphicRelaxation.h"
#include "cobordism/JointAction.h"

namespace tessera::cobordism {

/// # CovarianceRule
///
/// How the carried covariance is built from the carrier operator.
///
/// * `OccupiedProjector` — the spectral projector onto `occupiedModes` modes: a
///   Slater determinant, the quasi-free covariance of the Gaussian class.
/// * `BandFilling` — the spectrum (of \f$ h \f$, or of its group average under
///   a declared `bandSymmetry`) is grouped into bands of degenerate eigenvalues
///   (consecutive eigenvalues in the declared order within `bandTolerance` of
///   each other, relative to their size), and band \f$ b \f$ of rank
///   \f$ r_b \f$ carries the declared occupation \f$ n_b \f$ spread evenly
///   over it:
///   \f[ \Gamma=\sum_b \frac{n_b}{r_b}\,P_b , \f]
///   with \f$ P_b \f$ the band's spectral (Riesz) projector. This is the
///   one-body density of a many-body state that places \f$ n_b \f$ particles
///   in band \f$ b \f$ and is invariant under every symmetry that protects the
///   bands: by Schur's lemma such a density is a multiple of the projector on
///   each band that carries one irreducible representation, so the rule adds
///   no choice beyond the occupation numbers. It is the density the
///   whitepaper's certificates-blind backreaction reads a correlated state
///   through, the bilinear density \f$ \operatorname{tr}(\Gamma h) \f$, and it
///   is not idempotent when a band is partly filled; `purityDefect` reports
///   by how much.
///
/// Which bands carry the occupations away from the starting point is fixed by
/// `BandSelection`.
enum class CovarianceRule { OccupiedProjector, BandFilling };

/// # BandSelection
///
/// Where the occupied bands are chosen.
///
/// * `Continuation` (the default) — the bands are chosen once, at the point
///   the solve starts from, by the declared `OccupationOrder` (occupation
///   \f$ n_b \f$ goes to the \f$ b \f$-th band in that order), and at every
///   later point each occupied band is followed from its projector at the
///   previous point: it takes the \f$ r_b \f$ eigenvectors of the operator
///   whose weight in that projector,
///   \f$ w_k=\operatorname{Re}(\tilde v_k^{\mathsf T}P_b^{\rm prev}v_k) \f$, is
///   largest, with \f$ v_k \f$ an eigenvector and
///   \f$ \tilde v_k^{\mathsf T} \f$ the matching row of the inverse
///   eigenvector matrix. The bands are followed in the declared order and no
///   eigenvector is given to two of them. This is the continuation of each
///   band's contour: WP v17 line 151 selects a band of a non-normal operator
///   "by a closed contour in the complex spectral plane, not by sorting real
///   parts or imaginary parts". The overlap
///   \f$ \operatorname{tr}(P_bP_b^{\rm prev})/r_b \f$ of every band with its
///   previous projector is reported at every point, and so is any crossing,
///   a band whose modes no longer hold the places in the declared order they
///   held where the bands were chosen.
/// * `SortEveryIterate` — the bands are re-selected at every point by the
///   declared order, so a band that crosses another in the declared order
///   exchanges its occupation with it. The overlaps with the previous point's
///   bands are still reported, and an exchange shows as an overlap far below
///   one.
enum class BandSelection { Continuation, SortEveryIterate };

/// # The method
///
/// The fixed point is solved by Newton's method on the joint system,
/// written as \f$ F_{\rm sc}(z,U)=F(z,U,\Gamma(z,U))=0 \f$: the covariance
/// equation is solved exactly at every point the solve evaluates (the
/// covariance is rebuilt there), and the geometric equations are solved by
/// damped Newton steps with the Jacobian of \f$ F_{\rm sc} \f$
/// (`HolomorphicRelaxation` with a `CovarianceRebuild`). By the
/// Hellmann-Feynman identity \f$ F_{\rm sc} \f$ is the gradient of the
/// geometric action plus the occupied energy, and its Jacobian is their
/// Hessian.

/// # FiberConstraintForm
///
/// What the fiber constraints of a self-consistent solve pin
/// (`SelfConsistentMeanFieldDeclaration::fiberMoments`).
enum class FiberConstraintForm {
  /// The power sums \f$ p_j(h_{\mathcal C}) \f$, \f$ j=1,\ldots,m_{\rm c} \f$,
  /// of the occupied fiber: the constraints of WP v17 §3.4 as written.
  PowerSums,
  /// The eigenvalue of each occupied band,
  /// \f$ \lambda_b=\operatorname{tr}(P_bh)/r_b \f$
  /// (`SpectralConstraintForm::BandMean`), one constraint per occupied band
  /// in the declared order, the first \f$ m_{\rm c} \f$ of them. On a
  /// sheeted host every occupied band is one eigenvalue repeated once per
  /// sheet, so the \f$ r \f$ power sums carry only as many independent
  /// constraints as there are occupied bands; pinning the eigenvalues states
  /// those constraints without the dependent rows, and each band's projector
  /// is rebuilt at every point as the fiber's is.
  BandEigenvalues
};

/// # SelfConsistentMeanFieldDeclaration
///
/// The configuration of a self-consistent backreaction solve.
struct SelfConsistentMeanFieldDeclaration {
  /// How many modes of the carrier operator are filled under
  /// `CovarianceRule::OccupiedProjector`.
  std::size_t occupiedModes = 1;

  /// The order the modes and bands are counted in where the bands are chosen
  /// (at every point under `BandSelection::SortEveryIterate`).
  OccupationOrder occupationOrder = OccupationOrder::AscendingRealPart;

  /// How the covariance is built from the carrier operator. Under
  /// `BandFilling` `occupiedModes` is not read.
  CovarianceRule covarianceRule = CovarianceRule::OccupiedProjector;

  /// \f$ n_b \f$, the occupation of each band in the declared order, for
  /// `CovarianceRule::BandFilling`: entry \f$ b \f$ is the number of particles
  /// the \f$ b \f$-th band holds, at most its rank. Bands past the end of the
  /// vector are empty.
  std::vector<double> bandOccupations;

  /// The relative separation at or below which two consecutive ordered
  /// eigenvalues belong to one band under `CovarianceRule::BandFilling`:
  /// \f$ |\lambda_{i+1}-\lambda_i|\le\tau\max(1,|\lambda_i|) \f$.
  double bandTolerance = 1e-8;

  /// The declared symmetry the band rule reads its bands under, if any: the
  /// operators \f$ D(g) \f$ of a finite group acting on the carrier's cells,
  /// each flat row-major \f$ n\times n \f$. When present, the bands are those
  /// of the group average \f$ \bar h=|G|^{-1}\sum_g D(g)^{-1}hD(g) \f$ rather
  /// than of \f$ h \f$ itself. Such a \f$ \Gamma \f$ does not commute with
  /// \f$ h \f$ in general, so the matter term at fixed \f$ \Gamma \f$ is not
  /// gauge invariant and the rule is not gauge covariant. Empty, the default,
  /// reads the bands of \f$ h \f$: \f$ \Gamma \f$ is then a combination of
  /// Riesz projectors of \f$ h \f$ and commutes with it, which is the
  /// stationary pair of WP v17 §7 ("\f$ \Gamma^{*} \f$ a projector onto modes
  /// of \f$ h(z^{*}) \f$").
  std::vector<std::vector<std::complex<double>>> bandSymmetry;

  /// Where the occupied bands are chosen: once and then followed (the
  /// default), or re-selected by sorting at every point.
  BandSelection bandSelection = BandSelection::Continuation;

  /// \f$ m_{\rm c} \f$, the number of power sums of the occupied fiber the
  /// solve pins: the holomorphic spectral constraints of WP v17 §3.4, which
  /// controlled synthesis may impose to pin a carrier. The fiber is the
  /// occupied bands (their Riesz projector \f$ P_{\mathcal C} \f$, of rank
  /// \f$ r \f$ the sum of their ranks), chosen and followed as the
  /// covariance's bands are and rebuilt with them at every point, and the
  /// constraints are \f$ p_j(h_{\mathcal C})=p_j^{\star} \f$ for
  /// \f$ j=1,\ldots,m_{\rm c} \f$ with
  /// \f$ h_{\mathcal C}=P_{\mathcal C}hP_{\mathcal C}|_{\operatorname{Ran}
  /// P_{\mathcal C}} \f$ (`JointActionDeclaration::momentProjector`). Their
  /// complex multipliers \f$ \xi_j \f$ are unknowns of the solve beside the
  /// geometry, so the geometry declaration's multipliers are relaxed whatever
  /// it says; they start at the least-squares estimate at the starting point,
  /// the \f$ \xi \f$ that best balances the stationarity force there. Zero,
  /// the default, pins nothing. At most \f$ r \f$: the paper takes
  /// \f$ j=1,\ldots,r \f$. Under `FiberConstraintForm::BandEigenvalues`
  /// (`fiberConstraintForm`) it is instead the number of occupied bands
  /// whose eigenvalue is pinned, in the declared order, at most all of them.
  std::size_t fiberMoments = 0;

  /// The targets \f$ p_j^{\star} \f$ of the pinned power sums, one per
  /// pinned moment, in the operator's own unit. Empty (the default) takes the
  /// fiber's own values at the point the solve starts from, which pins the
  /// carrier as it was declared.
  std::vector<std::complex<double>> fiberMomentTargets;

  /// The unit \f$ s \f$ the pinned power sums are solved in
  /// (`JointActionDeclaration::momentScale`): the constraints are
  /// \f$ p_j(h_{\mathcal C}/s)=p_j^{\star}s^{-j} \f$, the same constraints,
  /// with gradients commensurate across \f$ j \f$. Zero (the default) takes
  /// the fiber's spectral radius at the starting point. The report gives
  /// targets, residuals and multipliers in the operator's own unit.
  double fiberMomentScale = 0.0;

  /// What the pinned constraints are: the fiber's power sums (the default)
  /// or its bands' eigenvalues. `fiberMomentTargets` are then the
  /// eigenvalues \f$ \lambda_b^{\star} \f$, one per pinned band.
  FiberConstraintForm fiberConstraintForm = FiberConstraintForm::PowerSums;

  /// The largest number of Newton steps of the joint system. Zero reads the
  /// starting point only.
  std::size_t maximumIterations = 24;

  /// The Euclidean norm of the stationarity force over the relaxed geometric
  /// fields at or below which the pair \f$ (z^{*},\Gamma^{*}) \f$ is declared
  /// self-consistent.
  ///
  /// The covariance is rebuilt at every point, so the covariance half of the
  /// fixed point holds exactly at every iterate and the force is the one
  /// condition; the covariance change between iterates is reported, and it
  /// measures the last step rather than a residual.
  double tolerance = 1e-9;

  /// The Newton solve of the geometry.
  ///
  /// Under `JointNewton` it declares the joint solve's step control: the
  /// Jacobian rule, the rank tolerance, the dampings, the guards (the zeros of
  /// \f$ W \f$, the held monopole sectors, the domain of the action), the
  /// shared coordinates and the length runaway ratio. The joint solve steps
  /// until its residual is at or below the smaller of this declaration's
  /// `tolerance` and the mean field's own `tolerance`, within the mean
  /// field's `maximumIterations`; this declaration's `maximumIterations` is
  /// not read.
  HolomorphicRelaxationDeclaration geometry;
};

/// # OccupiedBand
///
/// One occupied band at one point of a solve, as `BandFollower` selected it.
struct OccupiedBand {
  /// The band's place in the declared order at the point where the bands were
  /// chosen: the \f$ b \f$ of the occupation \f$ n_b \f$. Zero for the
  /// occupied modes of `CovarianceRule::OccupiedProjector`, which form one
  /// band.
  std::size_t declaredIndex = 0;
  /// \f$ n_b \f$, the number of particles the band holds.
  double occupation = 0.0;
  /// \f$ r_b \f$, the band's rank.
  std::size_t rank = 0;
  /// The band's eigenvalues at this point.
  std::vector<std::complex<double>> eigenvalues;
  /// The places of the band's modes in the declared order at this point,
  /// ascending.
  std::vector<std::size_t> positions;
  /// The places its modes held where the bands were chosen.
  std::vector<std::size_t> declaredPositions;
  /// \f$ \operatorname{tr}(P_bP_b^{\rm prev})/r_b \f$, the overlap of the
  /// band's Riesz projector with its projector at the previous point; one at
  /// the point where the bands were chosen. It is one when the band's modes
  /// did not change and falls when they change character; under
  /// `SortEveryIterate` a value far below one is an exchange of occupations
  /// between bands.
  std::complex<double> overlap{1.0, 0.0};
  /// Whether the band crossed another: its `positions` differ from its
  /// `declaredPositions`.
  bool crossed = false;
  /// Whether a mode of the band is degenerate, to the band tolerance, with a
  /// mode outside it, so that the band's projector depends on the basis the
  /// eigensolver chose inside that degenerate group.
  bool ambiguous = false;
  /// The band's Riesz projector at this point, \f$ V_bV_b^{-1} \f$ over its
  /// eigenvectors and the matching rows of the inverse eigenvector matrix,
  /// flat row-major.
  std::vector<std::complex<double>> projector;
};

/// # BandRead
///
/// What `BandFollower::read` builds from one operator.
struct BandRead {
  /// \f$ \Gamma \f$, flat row-major.
  std::vector<std::complex<double>> covariance;
  /// The ranks of the bands the spectrum groups into in the declared order
  /// under `CovarianceRule::BandFilling`; empty under `OccupiedProjector`.
  std::vector<std::size_t> ranks;
  /// The eigenvalues of the operator, in the declared order.
  std::vector<std::complex<double>> ordered;
  /// The occupied bands, in the declared order of the point where they were
  /// chosen.
  std::vector<OccupiedBand> bands;
  /// The eigenvalues of the occupied span in the declared order at this
  /// point: the declared number of modes under `OccupiedProjector`, and under
  /// `BandFilling` every mode up to the end of the last band with a nonzero
  /// occupation. This is the span a sort would occupy; the bands the
  /// covariance fills are `bands`, which differ from it after a crossing.
  std::vector<std::complex<double>> occupiedEigenvalues;
  /// \f$ |\lambda_{\rm first\ outside}-\lambda_{\rm last\ inside}| \f$ at the
  /// end of the occupied span in the declared order. Quiet NaN when every mode
  /// is in the span or none is.
  double spectralGap = 0.0;
  /// The smallest distance in the complex plane between an eigenvalue of an
  /// occupied band and an eigenvalue outside every occupied band: the room a
  /// contour around the occupied bands has. Quiet NaN when either set is
  /// empty.
  double bandIsolation = 0.0;
  /// Whether any occupied band crossed another (`OccupiedBand::crossed`).
  bool crossing = false;
  /// The smallest \f$ |\text{overlap}| \f$ over the occupied bands.
  double lowestOverlap = 1.0;
};

/// # BandFollower
///
/// The declared covariance rule together with its band selection. It builds
/// \f$ \Gamma \f$ from an operator, choosing the occupied bands by the
/// declared order at the first read and, under
/// `BandSelection::Continuation`, following them at every later read from the
/// reference that `follow` last set.
class BandFollower {
 public:
  /// @param declaration The covariance rule, the occupations, the band
  ///   tolerance, the declared order and the band selection; nothing else of
  ///   it is read.
  /// @throws std::invalid_argument under `BandFilling` when an occupation is
  ///   negative, when the occupations are empty or sum to zero, or when the
  ///   band tolerance is negative; under `OccupiedProjector` when no mode is
  ///   declared occupied.
  explicit BandFollower(const SelfConsistentMeanFieldDeclaration &declaration);

  /// The covariance of the operator \p operatorMatrix (flat row-major). The
  /// occupied bands are chosen by the declared order when no reference is set
  /// or under `SortEveryIterate`, and are followed from the reference
  /// otherwise. The reference is not moved.
  /// @throws std::invalid_argument when a declared band cannot hold its
  ///   occupation, when more band occupations are declared than the spectrum
  ///   has bands, when more modes are declared occupied than the operator has,
  ///   or when the operator is defective, so that no Riesz projector follows
  ///   from its eigenvectors.
  /// @throws std::runtime_error when the eigendecomposition does not
  ///   converge.
  [[nodiscard]] BandRead read(
      const std::vector<std::complex<double>> &operatorMatrix) const;

  /// Make the bands of \p read the reference the next `read` follows (and
  /// measures its overlaps against).
  void follow(const BandRead &read);

  /// Whether a reference is set.
  [[nodiscard]] bool following() const noexcept { return !reference_.empty(); }

 private:
  struct Reference {
    double occupation = 0.0;
    std::size_t rank = 0;
    std::size_t declaredIndex = 0;
    std::vector<std::size_t> declaredPositions;
    std::vector<std::complex<double>> projector;
  };

  CovarianceRule rule_;
  std::size_t occupiedModes_;
  std::vector<double> occupations_;
  double tolerance_;
  OccupationOrder order_;
  BandSelection selection_;
  std::vector<Reference> reference_;
};

/// # SelfConsistentMeanFieldStep
///
/// One iterate of a solve, recorded so a run can be read back. Iterate zero is
/// the point the solve starts from; each later iterate is the point one
/// Newton step of the joint system reached.
struct SelfConsistentMeanFieldStep {
  /// The iterate's index, counting from zero at the starting point.
  std::size_t iteration = 0;
  /// The Euclidean norm of the joint stationarity force, over the geometric
  /// fields the geometry declaration relaxes, measured with this iterate's
  /// covariance. A field held fixed contributes nothing, since its equation
  /// is not one the solve is asked to satisfy. Under declared edge classes
  /// (`HolomorphicRelaxationDeclaration::edgeClasses`) the fields are the
  /// shared coordinates and the force on each is the sum over the edges that
  /// carry it.
  double forceNorm = 0.0;
  /// Every term of the action at this iterate (`ActionTermRecord`), when the
  /// geometry declaration records terms; empty otherwise.
  std::vector<ActionTermRecord> terms;
  /// \f$ \lVert\Gamma_{n}-\Gamma_{n-1}\rVert_F \f$, the movement of the
  /// covariance from the previous iterate; at iterate zero, from the
  /// covariance the action was declared with (zero when it was declared
  /// empty).
  double covarianceChange = 0.0;
  /// \f$ \lVert\Gamma^2-\Gamma\rVert_F \f$ of this iterate's covariance.
  double purityDefect = 0.0;
  /// The complex action at this iterate; a quiet NaN when it is unavailable.
  std::complex<double> action{0.0, 0.0};
  /// Whether `action` could be evaluated (`JointAction::reportedValue`).
  bool actionAvailable = true;
  /// Why `action` is unavailable, by name; empty when it is available.
  std::string actionUnavailable;
  /// \f$ \operatorname{tr}(\Gamma h) \f$, the occupied energy, which for a
  /// spectral projector is the sum of the occupied eigenvalues.
  std::complex<double> occupiedEnergy{0.0, 0.0};
  /// The eigenvalues of the occupied span in the declared order
  /// (`BandRead::occupiedEigenvalues`).
  std::vector<std::complex<double>> occupiedEigenvalues;
  /// The gap at the end of the occupied span in the declared order
  /// (`BandRead::spectralGap`).
  double spectralGap = 0.0;
  /// The ranks of the bands the spectrum groups into under
  /// `CovarianceRule::BandFilling`, in the declared order; empty under
  /// `OccupiedProjector`.
  std::vector<std::size_t> bandRanks;
  /// The occupied bands as selected at this iterate, each with its overlap
  /// with its projector at the previous iterate and whether it crossed
  /// another.
  std::vector<OccupiedBand> bands;
  /// The distance separating the occupied bands from the rest of the
  /// spectrum (`BandRead::bandIsolation`).
  double bandIsolation = 0.0;
  /// Whether an occupied band had crossed another at this iterate.
  bool bandCrossing = false;
  /// The multipliers \f$ \xi_j \f$ of the pinned fiber moments at this
  /// iterate, in order of \f$ j \f$, in the operator's own unit (of the
  /// constraint \f$ \xi_j(p_j(h_{\mathcal C})-p_j^{\star}) \f$); empty when
  /// none is pinned.
  std::vector<std::complex<double>> multipliers;
  /// The Euclidean norm of the pinned constraints' residuals at this iterate,
  /// in the unit they are solved in (\f$ p_j(h_{\mathcal C}/s)-p_j^{\star}
  /// s^{-j} \f$, the equations the solve drives to zero); zero when none is
  /// pinned.
  double momentResidualNorm = 0.0;
  /// Whether the joint residual at this iterate is at or below the joint
  /// solve's tolerance.
  bool geometryConverged = false;
  /// The joint residual norm at this iterate.
  double geometryResidualNorm = 0.0;
  /// `RelaxationStop::Continued` at an iterate the solve stepped on from, and
  /// the reason the joint solve stopped at its last iterate.
  RelaxationStop geometryStopReason = RelaxationStop::Continued;
  /// The stop reason in words, with the numbers that decided it; empty for
  /// `Continued`.
  std::string geometryStopDetail;
  /// Whether a Newton iteration was taken from this iterate (a Jacobian
  /// formed and trial steps tried); its record is `newton`.
  bool newtonIterated = false;
  /// The Newton iteration taken from this iterate: the
  /// rank and the rank gap of the joint Jacobian, the damping and the length
  /// of the step, the guards that halved it, and whether a step was accepted.
  HolomorphicStep newton;
};

/// # SelfConsistentMeanFieldReport
///
/// What a self-consistent solve reached.
struct SelfConsistentMeanFieldReport {
  /// The band selection that ran.
  BandSelection bandSelection = BandSelection::Continuation;
  /// Every iterate, in order, from the starting point.
  std::vector<SelfConsistentMeanFieldStep> steps;
  /// The number of accepted Newton steps of the joint system.
  std::size_t iterations = 0;
  /// Whether the fixed-point conditions held at the declared tolerance.
  bool converged = false;
  /// Why the solve stopped (`relaxationStopName` gives its name).
  RelaxationStop stopReason = RelaxationStop::IterationBudget;
  /// The stop reason in words, with the numbers that decided it.
  std::string stopDetail;
  /// The stationarity force norm at the point the solve stopped at.
  double forceNorm = 0.0;
  /// The last movement of the covariance.
  double covarianceChange = 0.0;
  /// \f$ \lVert\Gamma^2-\Gamma\rVert_F \f$ at the point the solve stopped at.
  double purityDefect = 0.0;
  /// \f$ \Gamma^{*} \f$, flat row-major over the carrier degree's cells.
  std::vector<std::complex<double>> covariance;
  /// The eigenvalues of the occupied span in the declared order at
  /// \f$ z^{*} \f$.
  std::vector<std::complex<double>> occupiedEigenvalues;
  /// \f$ \operatorname{tr}(\Gamma^{*}h(z^{*})) \f$.
  std::complex<double> occupiedEnergy{0.0, 0.0};
  /// The gap at the end of the occupied span in the declared order at
  /// \f$ z^{*} \f$.
  double spectralGap = 0.0;
  /// The band ranks at \f$ z^{*} \f$ under `CovarianceRule::BandFilling`.
  std::vector<std::size_t> bandRanks;
  /// The occupied bands at \f$ z^{*} \f$.
  std::vector<OccupiedBand> bands;
  /// The distance separating the occupied bands from the rest of the
  /// spectrum at \f$ z^{*} \f$.
  double bandIsolation = 0.0;
  /// The number of iterates at which an occupied band had crossed another.
  std::size_t bandCrossingIterates = 0;
  /// The smallest \f$ |\text{overlap}| \f$ of an occupied band with its
  /// projector at the previous iterate, over every iterate.
  double lowestBandOverlap = 1.0;
  /// The complex action at \f$ (z^{*},\Gamma^{*}) \f$; a quiet NaN when it is
  /// unavailable.
  std::complex<double> action{0.0, 0.0};
  /// Whether `action` could be evaluated (`JointAction::reportedValue`).
  bool actionAvailable = true;
  /// Why `action` is unavailable, by name; empty when it is available.
  std::string actionUnavailable;
  /// The number of variables of the joint Jacobian.
  std::size_t jacobianSize = 0;
  /// The rank of the joint Jacobian (the Jacobian of the self-consistent
  /// force, the covariance rebuilt at every node) at the point the solve
  /// stopped at, decided at the geometry declaration's rank tolerance. The
  /// gauge directions are its expected null space.
  std::size_t jacobianRank = 0;
  /// Its largest singular value.
  double largestSingularValue = 0.0;
  /// Its smallest singular value counted in the rank.
  double smallestRetainedSingularValue = 0.0;
  /// Its largest singular value counted as zero; zero when none is.
  double largestDiscardedSingularValue = 0.0;
  /// The smallest retained over the largest discarded singular value, the
  /// joint Jacobian's rank gap; positive infinity when none is discarded.
  double rankGap = 0.0;
  /// The Kontsevich-Segal allowability margin of the geometry the solve
  /// stopped at (`HodgeLaplacian::kontsevichSegalMargin`): positive is
  /// allowable and negative is not. Bands are read on the allowable side
  /// (WP v17 line 151).
  double kontsevichSegalMargin = 0.0;
  /// \f$ r \f$, the rank of the occupied fiber (the sum of the occupied
  /// bands' ranks) where the bands were chosen.
  std::size_t fiberRank = 0;
  /// The unit \f$ s \f$ the pinned constraints were solved in.
  double momentScale = 1.0;
  /// What was pinned: the fiber's power sums or its bands' eigenvalues.
  FiberConstraintForm fiberConstraintForm = FiberConstraintForm::PowerSums;
  /// The targets of the pinned constraints, \f$ p_j^{\star} \f$ in order of
  /// \f$ j \f$ or \f$ \lambda_b^{\star} \f$ in the declared band order, in
  /// the operator's own unit; empty when none is pinned.
  std::vector<std::complex<double>> momentTargets;
  /// The multipliers \f$ \xi_j \f$ of the constraints
  /// \f$ \xi_j(p_j(h_{\mathcal C})-p_j^{\star}) \f$ at the point the solve
  /// stopped at, in the operator's own unit (\f$ s^{-j} \f$ times the
  /// multiplier solved for in the unit \f$ s \f$).
  std::vector<std::complex<double>> multipliers;
  /// \f$ p_j(h_{\mathcal C})-p_j^{\star} \f$ at the point the solve stopped
  /// at, in the operator's own unit.
  std::vector<std::complex<double>> momentResiduals;
  /// The Euclidean norm of the Hellmann-Feynman force at the point the solve
  /// stopped at: the carried state's own force
  /// \f$ \operatorname{tr}(\Gamma\,\partial h) \f$ on the relaxed geometric
  /// coordinates, without the geometric terms and without the constraints.
  double hellmannFeynmanForceNorm = 0.0;
  /// The moment-constrained action's Hessian on the range of the
  /// Hellmann-Feynman force (the condition of WP v17 line 265: self-trapping
  /// as a stationary point with an isolated band requires it positive): the
  /// Rayleigh quotient \f$ f^{\mathsf T}Hf/f^{\mathsf T}f \f$ of the geometric
  /// block \f$ H \f$ of the joint Jacobian at the point the solve stopped at
  /// (the Hessian of the geometric action, the occupied energy and the pinned
  /// moments' multiplier terms, the covariance and the fiber rebuilt at every
  /// node) along the Hellmann-Feynman force \f$ f \f$ projected onto the
  /// tangent space of the pinned constraints, both in the coordinates
  /// \f$ (z,\theta) \f$ with \f$ \delta=i\theta \f$ for a link. On the real
  /// slice (real squared lengths, unit-modulus face holonomies) the quotient
  /// is real and its sign is the condition's reading; away from it the
  /// quotient is complex and is reported as it is. Quiet NaN when the force
  /// vanishes on that tangent space or the joint Jacobian is unread.
  std::complex<double> forceHessian{0.0, 0.0};
  /// The scale of that Hessian, the Frobenius norm of the geometric block in
  /// the coordinates \f$ (z,\theta) \f$: the quotient's imaginary part is
  /// read against it, since the difference-rule Jacobian is accurate relative
  /// to its entries and a quotient can be small by cancellation.
  double forceHessianScale = 0.0;
  /// The largest \f$ |z_e| \f$ over the complex at the point the solve
  /// stopped at, over its value at the start.
  double largestLengthRatio = 1.0;
};

/// # SelfConsistentMeanField
///
/// The certificates-blind mean-field backreaction of Section 7 of the
/// whitepaper, solved to self-consistency.
///
/// Reference: Landau, "Über die Bewegung der Elektronen im Kristallgitter",
/// Physikalische Zeitschrift der Sowjetunion 3, 664 (1933); Pekar, "Local
/// quantum states of electrons in an ideal ion crystal", Zhurnal
/// Eksperimentalnoi i Teoreticheskoi Fiziki 16, 341 (1946) — the self-consistent
/// polaron this construction is the complex-first form of.
/// Reference: Bach, Lieb and Solovej, "Generalized Hartree-Fock theory and the
/// Hubbard model", Journal of Statistical Physics 76, 3 (1994), for the
/// quasi-free closure of a covariance coupled to its own mean field.
/// Reference: Ortega and Rheinboldt, "Iterative Solution of Nonlinear Equations
/// in Several Variables", SIAM Classics in Applied Mathematics 30, for the
/// damped Newton method the joint system is solved by.
///
/// The only channel from the carried state to the geometry is the bilinear
/// action density \f$ \tilde\psi^{\mathsf T}h(z,U)\psi \f$ evaluated on
/// \f$ \Gamma \f$ by Wick reduction, so the force on a geometric coordinate is
/// the Hellmann-Feynman term
/// \f$ \operatorname{tr}(\Gamma\,\partial h/\partial z_e) \f$ and the force on a
/// link is \f$ \operatorname{tr}(\Gamma\,U_e\,\partial h/\partial U_e) \f$.
/// Both are complex and neither is projected onto a real part. No cluster,
/// fiber, colour, exchange or baryon observable enters, which is the firewall
/// the sub-mode's name records.
///
/// A fixed point is the self-consistent polaron the whitepaper names:
/// \f$ \Gamma^{*} \f$ is the declared rule's density of the modes of
/// \f$ h(z^{*}) \f$, and the state's force balances the geometric action edge
/// by edge. It is a stationary point of a complex action rather than a
/// minimum of a real one, and the report carries the residuals that certify
/// it as such. It is solved for by Newton's method on the joint system, and
/// the declared `BandSelection` fixes which bands the covariance fills
/// (chosen once and followed, by default), which changes no equation.
///
/// A solve that finds no fixed point says why, by name
/// (`SelfConsistentMeanFieldReport::stopReason`): the iterations ran out; no
/// damped step reduced the residual; the smallest damped step changed a held
/// monopole number ("no stationary point in the declared monopole sector");
/// the smallest damped step left the domain of the action or came too close
/// to a zero of the Villain weight; the residual reached its floor on the
/// held set; or the squared lengths overflowed the double. The report also
/// carries the Kontsevich-Segal margin of the geometry the solve stopped at,
/// and the joint Jacobian's rank and rank gap there.
///
/// Under `OccupiedProjector` the covariance is idempotent at every iterate by
/// construction, and `purityDefect` measures that closure rather than
/// assuming it. Under `BandFilling` the covariance is the one-body density of
/// a correlated state and `purityDefect` measures its distance from a
/// projector.
class SelfConsistentMeanField {
 public:
  /// Build a solve over an action.
  ///
  /// @param action The joint action. The complex it refers to is the object
  ///   the solve writes. The covariance is rebuilt from the carrier operator
  ///   at every point, starting from the bands chosen at the initial
  ///   geometry, so a covariance declared on the action is replaced.
  /// @param declaration The occupation rule, the band selection and the
  ///   convergence controls.
  /// @throws std::invalid_argument when no mode is declared occupied (under
  ///   `BandFilling`, when the band occupations are empty, negative, or sum
  ///   to zero), which leaves the matter term identically zero and the
  ///   self-consistency empty, or when the band tolerance is negative.
  SelfConsistentMeanField(JointAction action,
                          SelfConsistentMeanFieldDeclaration declaration);

  /// Run the solve, writing the relaxed geometry into the complex as it goes.
  [[nodiscard]] SelfConsistentMeanFieldReport solve();

  /// The action, carrying the covariance and the multipliers as the solve left
  /// them.
  [[nodiscard]] const JointAction &action() const noexcept { return action_; }

 private:
  [[nodiscard]] SelfConsistentMeanFieldReport solveJointNewton();

  JointAction action_;
  SelfConsistentMeanFieldDeclaration declaration_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_SELFCONSISTENTMEANFIELD_H
