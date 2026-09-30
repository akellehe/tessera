// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_HOLOMORPHICRELAXATION_H
#define TESSERA_COBORDISM_HOLOMORPHICRELAXATION_H

#include <array>
#include <complex>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "cobordism/JointAction.h"

namespace tessera::cobordism {

/// # RelaxationStop
///
/// Why a solve stopped. Each reason is reported by name beside the
/// solve's numbers, so a solve that ended without a stationary point says
/// why rather than only reporting the residual it reached.
///
/// * `Converged` — the residual norm reached the declared tolerance.
/// * `IterationBudget` — the declared number of iterations ran out first.
/// * `NoDescent` — no damped Newton step reduced the residual norm: the
///   solve cannot move from the point it stopped at. The smallest trial step
///   was refused by the residual test.
/// * `SectorBoundary` — no stationary point in the declared monopole sector
///   along the Newton direction: the smallest trial step already changed a
///   held monopole number, so a held face holonomy is driven across
///   \f$ -1 \f$ (a Dirac string), where the monopole number read from the
///   principal arguments jumps.
/// * `DomainBoundary` — the smallest trial step reached a point at which the
///   action refuses to evaluate (a link driven to zero or infinity, a
///   singular operator, a face holonomy outside the domain of the holonomy
///   term) or its residual is not finite.
/// * `HolonomyZero` — the smallest trial step came within the declared
///   margin of a zero of the Villain weight \f$ W \f$.
/// * `HeldFloor` — with held sectors, the residual is at its floor on the
///   held set: the constrained Newton step would reduce the residual norm by
///   no more than the declared tolerance, because the complex equations
///   outnumber the real directions the held moduli leave free and the
///   linearized equations have no better solution on them. The residual that
///   remains is reported.
/// * `LengthRunaway` — the squared lengths ran off: an accepted step took the
///   largest \f$ |z_e| \f$ beyond the declared multiple of its value at the
///   start of the solve
///   (`HolomorphicRelaxationDeclaration::lengthRunawayRatio`).
///   The linear stiffness stand-in's force on \f$ z_e \f$,
///   \f$ (1/2\kappa)(1-\ell_{0,e}/\ell_e) \f$, tends to the constant
///   \f$ 1/2\kappa \f$ as \f$ |z_e| \f$ grows while a mode's force decays,
///   so the residual norm has a plateau at infinite length that the monotone
///   residual test accepts; this is how that runaway is recognised.
/// * `NoProgress` — an outer iteration of the alternation of
///   `SelfConsistentMeanField` made no progress: its inner solve accepted no
///   step and re-occupation left the covariance unchanged, so a further
///   iteration would repeat it exactly.
/// * `Continued` — not a stop: in a per-iterate trace, the solve accepted a
///   step from this iterate and went on.
enum class RelaxationStop {
  Converged,
  IterationBudget,
  NoDescent,
  SectorBoundary,
  DomainBoundary,
  HolonomyZero,
  HeldFloor,
  LengthRunaway,
  NoProgress,
  Continued
};

/// The name of a stop reason as the reports print it, for example
/// "no stationary point in the declared monopole sector" for
/// `RelaxationStop::SectorBoundary`.
[[nodiscard]] std::string relaxationStopName(RelaxationStop reason);

/// # RebuiltCarrierState
///
/// What a `CovarianceRebuild` sets on the action at one point: the carried
/// covariance \f$ \Gamma \f$ and, when the solve imposes spectral constraints
/// on a fiber, the fiber's Riesz projector
/// (`JointActionDeclaration::momentProjector`), both flat row-major over the
/// carrier's cells.
struct RebuiltCarrierState {
  std::vector<std::complex<double>> covariance;
  /// Empty when the solve imposes no constraint on a fiber.
  std::vector<std::complex<double>> momentProjector;
  /// The band projectors of the solve's band-mean constraints
  /// (`JointActionDeclaration::momentBandProjectors`), in the constraints'
  /// order; empty when it imposes none.
  std::vector<std::vector<std::complex<double>>> bandProjectors;
};

/// # CovarianceRebuild
///
/// The rule by which a self-consistent solve rebuilds the carried covariance
/// \f$ \Gamma \f$ (and the constrained fiber's projector, when one is
/// declared) from the carrier operator at every point it evaluates the
/// residual at. With it `HolomorphicRelaxation` solves the self-consistent
/// system \f$ F_{\rm sc}(z,U)=F(z,U,\Gamma(z,U))=0 \f$ instead of the
/// stationarity at a fixed \f$ \Gamma \f$: every residual, every Jacobian
/// column and every trial point reads \f$ \Gamma \f$ rebuilt there. The
/// equations are the same; only \f$ \Gamma \f$ is no longer held.
struct CovarianceRebuild {
  /// The state at the geometry the action currently refers to. Called with
  /// the geometry moved to each point the solve evaluates, and required to
  /// leave the geometry as it found it.
  std::function<RebuiltCarrierState(const JointAction &)> at;
  /// Called once after every accepted step, with the action at the accepted
  /// point and \f$ \Gamma \f$ already rebuilt there, so a rule that follows
  /// the carrier's bands from point to point can move its reference.
  std::function<void(const JointAction &)> accepted;
};

/// # HolomorphicJacobianMode
///
/// How the Jacobian of the stationarity system is formed. The residual is an
/// exact analytic function of the variables in either case; this selects only
/// how its derivative is obtained.
///
/// * `ContourDerivative` — the Cauchy derivative on a small circle,
///   \f$ \partial F/\partial v \approx \frac{1}{m\rho}\sum_{n=0}^{m-1}
///       F(v+\rho\,\omega^{n})\,\omega^{-n} \f$ with \f$ \omega=e^{2\pi i/m} \f$.
///   Because the residual is holomorphic, the trapezoidal rule on the circle
///   converges geometrically in the node count: the first term it misses is of
///   order \f$ \rho^{m} \f$ times the \f$ (m{+}1) \f$-st Taylor coefficient, so
///   the default eight nodes at a relative radius of \f$ 10^{-2} \f$ leave a
///   truncation of order \f$ 10^{-16} \f$ and the derivative is exact to
///   rounding. This is the default, and it is the mode that makes holomorphy
///   pay.
///   The rule reads one branch of the residual over the whole circle, so it is
///   the right rule exactly when the residual is analytic on the whole disc.
/// * `RealAxisDifference` — the two-node rule
///   \f$ (F(v+\rho)-F(v-\rho))/(2\rho) \f$ with both nodes placed exactly on
///   the real axis. For a residual that is analytic on the disc this is the
///   \f$ m=2 \f$ case of the same contour and carries an
///   \f$ O(\rho^{2}) \f$ truncation, so a caller declaring it usually declares
///   a smaller radius with it.
///
///   Its purpose is a residual that is analytic on each side of a cut along the
///   real axis but not across it. The dual Lorentzian Regge action's exact
///   gradient is such a residual: the deficit angle is taken on the principal
///   branch with no Riemann-sheet label carried, so an arbitrarily small
///   positive imaginary part in a squared length shifts a hinge's deficit by
///   \f$ 2\pi \f$ and its contribution to the gradient by \f$ 2\pi \f$ times
///   the hinge's dual volume. A contour around a real configuration crosses
///   that cut and reads two sheets; two nodes on the axis stay on one. Both
///   nodes are constructed as exact real numbers rather than as
///   \f$ \rho\,e^{i\pi n} \f$, whose sine is not exactly zero in binary
///   floating point and would put one node on the far side of the cut.
enum class HolomorphicJacobianMode { ContourDerivative, RealAxisDifference };

/// # HeldMonopoleSector
///
/// One declared cluster whose monopole sector is boundary data of the solve:
/// the outward-oriented faces of its bounding cut, each given by its three
/// vertex ids in the order that orients it outward, and the declared monopole
/// number \f$ \mu \f$, the sum of the principal arguments of the outward face
/// holonomies over \f$ 2\pi \f$ (read on their \f$ U(1) \f$ parts).
struct HeldMonopoleSector {
  std::vector<std::array<std::uint64_t, 3>> faces;
  int monopoleNumber = 0;
};

/// # HolomorphicRelaxationDeclaration
///
/// The configuration of a holomorphic Newton solve. Every field is a numerical
/// control of the root find; none of them changes which equations are solved.
struct HolomorphicRelaxationDeclaration {
  /// Whether the squared lengths \f$ z_e \f$ are variables of the solve. When
  /// false they are held at their current values and the length stationarity
  /// equations are dropped from the system, because requiring an equation of a
  /// frozen variable would make the system overdetermined rather than partially
  /// relaxed.
  bool relaxLengths = true;

  /// Whether the links \f$ U_e \f$ are variables of the solve, under the same
  /// rule as `relaxLengths`.
  bool relaxLinks = true;

  /// Whether the multipliers \f$ \xi_j \f$ are variables of the solve. When
  /// false they are held at their declared values and the moment equations are
  /// dropped, which turns a constrained solve into an unconstrained one at a
  /// fixed external source.
  bool relaxMultipliers = true;

  /// The largest number of Newton iterations taken before the solve reports
  /// what it reached.
  std::size_t maximumIterations = 24;

  /// The Euclidean norm of the residual vector at or below which the solve is
  /// declared converged. This is a convergence certificate on the complex
  /// equations, not a functional minimized in their place.
  double tolerance = 1e-10;

  /// The number of nodes \f$ m \f$ on the contour, for `ContourDerivative`.
  /// Must be at least five, so that no Jacobian in this solver rests on a
  /// truncation shorter than the repository's floor for a series.
  std::size_t contourNodes = 8;

  /// The radius \f$ \rho \f$ of the contour, relative to the magnitude of the
  /// coordinate being differentiated and floored at one, so that a coordinate
  /// of very different scale is still differentiated on a circle inside its own
  /// domain of analyticity.
  double contourRadius = 1e-2;

  /// The largest number of step halvings tried when a full Newton step does not
  /// reduce the residual norm. Damping is a globalization of the root find and
  /// leaves the equations being solved unchanged.
  std::size_t maximumDampings = 16;

  /// How the Jacobian is formed.
  HolomorphicJacobianMode jacobianMode =
      HolomorphicJacobianMode::ContourDerivative;

  /// The relative threshold below which a singular value of the Jacobian counts
  /// as zero in the minimum-norm solve of the Newton system: the rank is the
  /// number of singular values \f$ \sigma_i>\tau\,\sigma_{\max} \f$, decided
  /// on the singular values themselves and not on a pivoted-QR diagonal,
  /// whose magnitudes only bracket them.
  double rankTolerance = 1e-12;

  /// Whether every recorded step, and the starting point, carries every
  /// term of the action with its value and gradient norm
  /// (`ActionTermRecord`, `actionTermRecords`). Off by default: it costs a
  /// few stationarity evaluations per step, and changes no step.
  bool recordTerms = false;

  /// Coordinates shared by several edges. Empty (the default) makes every
  /// edge its own coordinate. Otherwise entry \f$ e \f$, one per edge in
  /// `getEdgeList()` order, is the index of the shared coordinate edge
  /// \f$ e \f$ carries, the indices running over \f$ 0,\dots,K-1 \f$ with
  /// every one used. The edges of one class carry one squared length and
  /// one link (on the orientation `edgeClassOrientations` relates them to),
  /// the solve's variables are one squared length and one link per class,
  /// and the equation of a class is the sum of its edges' stationarity
  /// equations, which is the derivative of the whole action along the shared
  /// coordinate. This is how a \f$ k \f$-sheeted support is relaxed as one
  /// base field (WP v17 §8, "Sheet convention (adopted)": equal squared
  /// lengths and equal connection values on corresponding edges): every
  /// member of a class is written the same value, so identical sheets stay
  /// identical exactly. The members of a class must carry equal fields when
  /// the solve starts.
  std::vector<std::size_t> edgeClasses;

  /// With `edgeClasses`, the orientation of each edge relative to its class:
  /// \f$ +1 \f$ when the edge's stored link is the class's link, \f$ -1 \f$
  /// when it is its inverse. Empty means \f$ +1 \f$ for every edge.
  std::vector<int> edgeClassOrientations;

  /// The declared clearance of the face holonomies from the zeros of the
  /// Villain weight \f$ W \f$: a trial step whose multiplicative path brings
  /// any face holonomy within this relative distance of a zero
  /// (`JointAction::holonomyZeroClearance`, sampled at a quarter of this
  /// spacing) is halved, as a step that does not reduce the residual is. The
  /// potential \f$ -\beta_V\log W \f$ and its derivatives are singular at a
  /// zero, so a step onto or past one leaves the domain the equations are
  /// posed on. This is step control only: the equations are unchanged. It has
  /// no effect under the Wilson form, whose potential has no singularity.
  double holonomyZeroMargin = 0.05;

  /// The monopole sectors held as boundary data (controlled synthesis, WP v18
  /// Section 11.1). The monopole number through a declared cluster's bounding
  /// cut is boundary data, held on that cut by this rule; it is not an
  /// invariant of continuous relaxation, because a face flux is defined
  /// modulo \f$ 2\pi \f$ and the number changes by \f$ \pm1 \f$ whenever a
  /// face holonomy on the cut passes through \f$ -1 \f$, so an odd sector
  /// persists only where it is held or where the dynamics keeps the cut's
  /// holonomies away from \f$ -1 \f$; in the bulk, and on every grown level,
  /// it is read and never held. For every declared sector the solve keeps (i)
  /// the modulus of every
  /// face holonomy on its cut, so a holonomy on the unit circle stays on it,
  /// and (ii) the monopole number, by halving any trial step after which a
  /// sector reads a different number. The arguments of the face holonomies
  /// and every other connection degree of freedom relax freely. The declared
  /// numbers must be the ones the starting configuration carries.
  ///
  /// The held moduli are kept by the Newton step itself. In the
  /// multiplicative link coordinates \f$ U_e\mapsto U_ee^{\delta_e} \f$ the
  /// logarithm of a held face's modulus moves by
  /// \f$ \sum_e\epsilon_{\tau e}\operatorname{Re}\delta_e \f$, which is linear,
  /// so the held set is the linear subspace
  /// \f$ C\operatorname{Re}\delta=0 \f$ (with \f$ C \f$ the held faces'
  /// coboundary rows) and it is its own tangent space. The step is the
  /// minimum-norm least-squares solution of the linearized equations
  /// \f$ J d=-F \f$ over that subspace: a real least-squares problem in the
  /// real and imaginary parts of the length and multiplier steps, the link
  /// phases \f$ \operatorname{Im}\delta \f$, and the link moduli along the
  /// kernel of \f$ C \f$, with its rank decided at the Jacobian's own rank
  /// boundary (`HolomorphicStep::constrainedRank`). It is the constrained
  /// Newton (Gauss-Newton) step, and so a descent direction for the residual
  /// norm whenever it is not zero. The complex equations can outnumber the
  /// real directions the held set leaves free, so the step's own linearized
  /// residual (`HolomorphicStep::linearResidual`) need not vanish; it is
  /// reported.
  std::vector<HeldMonopoleSector> heldSectors;

  /// The declared bound on the growth of the squared lengths: when an
  /// accepted step takes the largest \f$ |z_e| \f$ (over the length
  /// coordinates) beyond this multiple of its value at the start of the
  /// solve, the solve stops and reports `RelaxationStop::LengthRunaway`. The
  /// linear stiffness stand-in's force on a squared length saturates at
  /// \f$ 1/2\kappa \f$ as the length grows, so the residual norm has a plateau
  /// at infinite length that the monotone residual test would otherwise
  /// accept step after step. This is a stop, not a change of the equations.
  /// Zero or a non-finite value disables it.
  double lengthRunawayRatio = 1e2;
};

/// # ActionTermRecord
///
/// One term of the joint action at a recorded point of a relaxation
/// (`JointAction::termGradients`): its value and the Euclidean norm of its
/// stationarity gradient on the coordinates the relaxation relaxes, reduced
/// onto the declared edge classes as the residual is. The multiplier
/// equations are constraints rather than gradients and are left out, as they
/// are of the force norm.
struct ActionTermRecord {
  /// As `ActionTermGradient::name`, plus `"constraints"` for the sum of the
  /// constraint terms and `"action"` for the whole action.
  std::string name;
  /// As `ActionTermGradient::label`: `"sum_j xi_j (c_j - c_j*)"` and `"S"`
  /// for the two sums.
  std::string label;
  std::complex<double> weight{0.0, 0.0};
  std::complex<double> bare{0.0, 0.0};
  bool factored = true;
  std::complex<double> value{0.0, 0.0};
  /// The Euclidean norm of the term's stationarity on the relaxed
  /// coordinates; for `"action"`, of the whole stationarity, which is the
  /// force norm the solve reports.
  double gradientNorm = 0.0;
};

/// # HolomorphicStep
///
/// One Newton iteration, recorded so a run can be read back rather than only
/// its outcome.
struct HolomorphicStep {
  /// The iteration index, counting from zero.
  std::size_t iteration = 0;
  /// The Euclidean norm of the stationarity residual at the point the step was
  /// taken from.
  double residualNorm = 0.0;
  /// The Euclidean norm of the accepted displacement in the variables.
  double stepNorm = 0.0;
  /// The damping factor the accepted step carried: one for a full Newton step,
  /// and a negative power of two when the full step did not reduce the residual.
  double damping = 1.0;
  /// The rank of the Jacobian at this point: the number of its singular
  /// values above `rankTolerance` times the largest. It is below the variable
  /// count whenever the connection is relaxed, because the action is gauge
  /// invariant.
  std::size_t jacobianRank = 0;
  /// The relative rank tolerance the rank was decided at.
  double rankTolerance = 0.0;
  /// The largest singular value of the Jacobian.
  double largestSingularValue = 0.0;
  /// The smallest singular value counted in the rank; NaN when the rank is
  /// zero.
  double smallestRetainedSingularValue = 0.0;
  /// The largest singular value counted as zero; zero when none is.
  double largestDiscardedSingularValue = 0.0;
  /// The rank gap: the smallest retained over the largest discarded singular
  /// value, positive infinity when none is discarded. A gap near one says the
  /// rank decision is not separated from the spectrum's continuation.
  double rankGap = 0.0;
  /// The complex action \f$ S(z,U,\Gamma) \f$ at the point the step was taken
  /// from.
  std::complex<double> action{0.0, 0.0};
  /// How many of this iteration's step halvings the holonomy zero guard forced
  /// (`HolomorphicRelaxationDeclaration::holonomyZeroMargin`), as opposed to
  /// the residual test.
  std::size_t zeroGuardDampings = 0;
  /// The smallest relative distance of a face holonomy to a zero of \f$ W \f$
  /// at the point the step was taken from; positive infinity when the declared
  /// holonomy term has no zero.
  double holonomyZeroDistance = 0.0;
  /// How many of this iteration's step halvings the held monopole sectors
  /// forced (a trial step that changed a declared monopole number).
  std::size_t sectorGuardDampings = 0;
  /// How many of this iteration's step halvings a trial point outside the
  /// action's domain forced: a point at which the action refuses to evaluate
  /// (a link driven to zero or infinity) or its residual is not finite.
  std::size_t domainGuardDampings = 0;
  /// How many of this iteration's step halvings the residual test forced (a
  /// trial point whose residual norm was not below the current one).
  std::size_t residualTestDampings = 0;
  /// Whether a damped step was accepted. When none was, the solve stopped at
  /// the point this step was taken from, and `damping` and `stepNorm` are
  /// zero.
  bool accepted = false;
  /// Every term of the action at the point the step ended at, with its value
  /// and gradient norm (`ActionTermRecord`); filled when the declaration's
  /// `recordTerms` is set, empty otherwise.
  std::vector<ActionTermRecord> terms;
  /// \f$ \lVert F+Jd\rVert/\lVert F\rVert \f$ for the full Newton step
  /// \f$ d \f$: how much of the residual the linearized equations leave. It is
  /// at rounding level for an unconstrained step on a Jacobian of full rank
  /// in the physical directions, and it is the constrained step's own floor
  /// when held sectors restrict the step.
  double linearResidual = 0.0;
  /// Whether the step was solved on the tangent space of held sectors
  /// (`HolomorphicRelaxationDeclaration::heldSectors`) rather than on the
  /// whole space.
  bool constrainedStep = false;
  /// For a constrained step, the rank of the real least-squares system it was
  /// solved from; zero otherwise. It is decided at the Jacobian's rank
  /// boundary, the geometric mean of the Jacobian's smallest retained
  /// singular value and the larger of its largest discarded one and its cut:
  /// a singular value of the constrained system below it comes from a
  /// direction of the held set in the Jacobian's numerical null space (a
  /// gauge direction), since the constrained system's singular values are
  /// bounded below by the Jacobian's smallest one.
  std::size_t constrainedRank = 0;
  /// For a constrained step, that system's smallest retained over its largest
  /// discarded singular value (positive infinity when none is discarded); NaN
  /// otherwise.
  double constrainedRankGap = 0.0;
  /// Whether `action` could be evaluated (`JointAction::reportedValue`).
  bool actionAvailable = true;
  /// Why `action` is unavailable, by name; empty when it is available.
  std::string actionUnavailable;
};

/// # HolomorphicRelaxationReport
///
/// What a solve reached, and the trace of how it got there.
struct HolomorphicRelaxationReport {
  /// Every iteration, in order.
  std::vector<HolomorphicStep> steps;
  /// Whether the residual norm reached the declared tolerance.
  bool converged = false;
  /// The residual norm at the starting point.
  double initialResidualNorm = 0.0;
  /// Every term of the action at the starting point (`ActionTermRecord`);
  /// filled when the declaration's `recordTerms` is set.
  std::vector<ActionTermRecord> initialTerms;
  /// The residual norm at the point the solve stopped at.
  double residualNorm = 0.0;
  /// The complex action at the point the solve stopped at.
  std::complex<double> action{0.0, 0.0};
  /// The multipliers \f$ \xi_j \f$ at the point the solve stopped at, in the
  /// declaration order of the constraints.
  std::vector<std::complex<double>> multipliers;
  /// \f$ p_j(h)-p_j^{\star} \f$ at the point the solve stopped at.
  std::vector<std::complex<double>> momentResiduals;
  /// The number of iterations whose step the holonomy zero guard damped at
  /// least once.
  std::size_t zeroGuardDampedSteps = 0;
  /// The number of iterations whose step the held monopole sectors damped at
  /// least once.
  std::size_t sectorGuardDampedSteps = 0;
  /// The monopole number of every declared sector at the point the solve
  /// stopped, in declaration order.
  std::vector<int> sectorMonopoleNumbers;
  /// \f$ \max_f |\log|F_f|_{\rm end} - \log|F_f|_{\rm start}| \f$ over the
  /// held faces: the drift of the held moduli, zero to rounding.
  double heldModulusDrift = 0.0;
  /// The number of hinges the primal Regge sum runs over
  /// (`JointAction::reggeHingeCount`).
  std::size_t reggeHingeCount = 0;
  /// True when a declared primal Regge term has no hinge on this complex under
  /// the declared hinge rule, so that it and its gradient were identically zero
  /// throughout the solve (`JointAction::reggeStructurallyZero`). The solve
  /// then relaxed the lengths without any Regge term.
  bool reggeStructurallyZero = false;
  /// The number of dihedral angles of the primal Regge sum whose continued
  /// sheet differs from the principal one at the point the solve stopped at
  /// (`JointAction::reggeOffPrincipalAngles`).
  std::size_t reggeOffPrincipalAngles = 0;
  /// Why the solve stopped.
  RelaxationStop stopReason = RelaxationStop::IterationBudget;
  /// The stop reason in words, with the numbers that decided it.
  std::string stopDetail;
  /// The largest \f$ |z_e| \f$ over the length coordinates at the point the
  /// solve stopped at, over its value at the start; one when the lengths are
  /// not relaxed.
  double largestLengthRatio = 1.0;
  /// Whether `action` could be evaluated (`JointAction::reportedValue`).
  bool actionAvailable = true;
  /// Why `action` is unavailable, by name; empty when it is available.
  std::string actionUnavailable;
};

/// # HolomorphicRelaxation
///
/// A Newton root find on the holomorphic stationarity equations of
/// `JointAction`,
/// \f[
///   \frac{\partial S}{\partial z_e}=0,\qquad
///   U_e\frac{\partial S}{\partial U_e}=0,\qquad
///   p_j(h)-p_j^{\star}=0 .
/// \f]
///
/// Reference: Ortega and Rheinboldt, "Iterative Solution of Nonlinear Equations
/// in Several Variables", SIAM Classics in Applied Mathematics 30, for the
/// damped Newton method and its convergence.
/// Reference: Lyness and Moler, "Numerical differentiation of analytic
/// functions", SIAM Journal on Numerical Analysis 4, 202 (1967), for the
/// contour rule the Jacobian is formed by.
///
/// This solves the equations themselves. It does not minimize the residual
/// norm, it does not minimize a real part, and it does not select a real
/// projection of the action: the residual norm appears only as the quantity the
/// damping compares and as the convergence certificate the report carries. Both
/// the real and the imaginary part of every equation are driven to zero because
/// the equation is one complex equation, and a Newton step in the complex
/// variables solves both at once.
///
/// ## The variables and how they are updated
///
/// The variables are the complex squared lengths \f$ z_e=\ell_e^2 \f$, the
/// Maurer-Cartan increments \f$ \delta_e \f$ of the links on each edge's stored
/// orientation, and the multipliers \f$ \xi_j \f$.
///
/// The connection is updated multiplicatively: a step \f$ \delta_e \f$ sends
/// \f$ U_e \mapsto U_e\,e^{\delta_e} \f$. The mesh stores the connection as the
/// phase \f$ \varphi_e \f$ of \f$ U_e=e^{i\varphi_e} \f$, and the identical
/// update on that storage is \f$ \varphi_e \mapsto \varphi_e-i\,\delta_e \f$,
/// which is an exact rewriting of the multiplicative step and not a choice of
/// logarithm branch: no logarithm of a link is ever formed, and the increment
/// is applied to the stored coordinate rather than recovered from the product.
///
/// The squared length is written back through `mesh::Edge::setLength`, which
/// takes \f$ \ell \f$ rather than \f$ \ell^2 \f$. The root is chosen by
/// continuation from the edge's current length: of the two square roots of the
/// new \f$ z_e \f$, the solver keeps the one on the same side as the current
/// \f$ \ell_e \f$, so a relaxation path never jumps between the two sheets of
/// \f$ \ell\mapsto\ell^2 \f$ between consecutive iterations.
///
/// With `HolomorphicRelaxationDeclaration::edgeClasses` the variables are
/// shared coordinates, each written to every edge of its class, and each
/// equation is the sum of the stationarity equations of its class's edges.
///
/// ## Why the Newton system is solved in the minimum-norm sense
///
/// The action is gauge invariant, so its link stationarity vector is orthogonal
/// to every pure-gauge direction and the connection block of the Jacobian is
/// singular along all of them — one null direction per vertex, less one per
/// connected component. That is a property of the theory, not a defect of the
/// discretization, and a solver that inverted the Jacobian would be inverting a
/// singular matrix. The step is therefore the minimum-norm least-squares
/// solution from the singular value decomposition, with the singular values
/// at or below `rankTolerance` times the largest counted as zero, which is the
/// unique solution orthogonal to the numerical null space: the connection
/// moves only in physical directions and the gauge is left where it was. The
/// rank and the gap at the rank decision are recorded on every step. With
/// held sectors the same minimum-norm rule is applied to the real
/// least-squares system over the held set's tangent space
/// (`HolomorphicRelaxationDeclaration::heldSectors`).
///
/// ## Step control and why a solve stops
///
/// A full Newton step that does not reduce the residual norm is halved, up to
/// `maximumDampings` times, and so is a trial step that the declared guards
/// refuse: one that comes too close to a zero of the Villain weight, one that
/// changes a held monopole number, and one that reaches a point at which the
/// action refuses to evaluate. Damping is a globalization and changes no
/// equation. When no trial step is accepted the solve stops, and the report
/// names the reason from the refusal of the smallest trial step
/// (`RelaxationStop`); a runaway of the squared lengths also stops it by name.
///
/// ## The self-consistent system
///
/// Built with a `CovarianceRebuild`, the solve rebuilds \f$ \Gamma \f$ from
/// the carrier operator at every point it evaluates, so it solves the joint
/// system \f$ F_{\rm sc}(z,U)=0 \f$ of `SelfConsistentMeanField` with the
/// Jacobian \f$ H_{\rm sc} \f$ of the self-consistent force. The equations
/// are those of the joint action; only the order in which they are solved
/// differs from holding \f$ \Gamma \f$ fixed.
/// Every term of \p action with its value and the norm of its stationarity
/// gradient on the coordinates \p declaration relaxes (`ActionTermRecord`),
/// the per-edge gradients summed over the declared edge classes as the
/// relaxation's residual is; then the sum of the constraint terms
/// (`"constraints"`) and the whole action (`"action"`), each with the norm of
/// its summed gradient.
[[nodiscard]] std::vector<ActionTermRecord> actionTermRecords(
    const JointAction &action,
    const HolomorphicRelaxationDeclaration &declaration);

class HolomorphicRelaxation {
 public:
  /// Build a solve over an action.
  ///
  /// @param action The action whose stationarity equations are solved. The
  ///   instance is held by value and its multipliers are advanced by the solve;
  ///   the complex it refers to is the object whose edge lengths and phases the
  ///   solve writes.
  /// @param declaration The numerical controls of the root find.
  /// @param rebuild Empty (the default) to hold the action's covariance fixed;
  ///   otherwise the rule that rebuilds \f$ \Gamma \f$ at every point the
  ///   solve evaluates.
  /// @throws std::invalid_argument when the declared contour node count is
  ///   below five, when the contour radius is not positive, when no field is
  ///   declared relaxable, or when declared edge classes do not cover the
  ///   edges, skip a class index, carry an orientation other than plus or
  ///   minus one, or start with unequal fields inside a class.
  HolomorphicRelaxation(JointAction action,
                        HolomorphicRelaxationDeclaration declaration,
                        CovarianceRebuild rebuild = {});

  /// Run the solve, writing the relaxed fields into the complex as it goes.
  [[nodiscard]] HolomorphicRelaxationReport solve();

  /// The action, carrying the multipliers as the solve left them.
  [[nodiscard]] const JointAction &action() const noexcept { return action_; }

  /// The Jacobian of the stationarity system at the current point, flat
  /// row-major, with the rows in the residual's block order and the columns in
  /// the variable order — the relaxed length coordinates, then the relaxed link
  /// coordinates, then the relaxed multipliers. Under declared edge classes a
  /// length or link coordinate is one class, in class order. Exposed so that the derivative
  /// the solve steps along can be inspected and checked against an independent
  /// evaluation rather than only trusted. With a `CovarianceRebuild` it is
  /// the Jacobian of the self-consistent force, \f$ \Gamma \f$ rebuilt at
  /// every node.
  ///
  /// Forming it moves each coordinate around its own contour in turn and
  /// restores the complex exactly after every evaluation, so the geometry is
  /// the same before and after the call.
  [[nodiscard]] std::vector<std::complex<double>> jacobian() const;

  /// The residual of the equations in scope at the current point, in the
  /// block order of `jacobian`'s rows, with \f$ \Gamma \f$ rebuilt first when
  /// a `CovarianceRebuild` is declared.
  [[nodiscard]] std::vector<std::complex<double>> residual() const;

  /// The number of rows of `jacobian`, which is the number of equations in
  /// scope.
  [[nodiscard]] std::size_t equationCount() const;

  /// The number of columns of `jacobian`, which is the number of variables in
  /// scope.
  [[nodiscard]] std::size_t variableCount() const;

 private:
  JointAction action_;
  HolomorphicRelaxationDeclaration declaration_;
  CovarianceRebuild rebuild_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_HOLOMORPHICRELAXATION_H
