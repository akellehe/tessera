// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_HOLOMORPHICRELAXATION_H
#define TESSERA_COBORDISM_HOLOMORPHICRELAXATION_H

#include <array>
#include <complex>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "cobordism/JointAction.h"

namespace tessera::cobordism {

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
  /// as zero in the minimum-norm solve of the Newton system.
  double rankTolerance = 1e-12;

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

  /// The monopole sectors held as boundary data (controlled synthesis: odd
  /// sectors are superselection data set by boundary or initial conditions,
  /// WP §9). For every declared sector the solve keeps (i) the modulus of every
  /// face holonomy on its cut, by removing from each link step the part of its
  /// real (modulus) component that would change those moduli, so a holonomy
  /// on the unit circle stays on it, and (ii) the monopole number, by halving
  /// any trial step after which a sector reads a different number. The
  /// arguments of the face holonomies and every other connection degree of
  /// freedom relax freely. The declared numbers must be the ones the starting
  /// configuration carries.
  std::vector<HeldMonopoleSector> heldSectors;
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
  /// The rank of the Jacobian at this point. It is below the variable count
  /// whenever the connection is relaxed, because the action is gauge invariant.
  std::size_t jacobianRank = 0;
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
/// ## Why the Newton system is solved in the minimum-norm sense
///
/// The action is gauge invariant, so its link stationarity vector is orthogonal
/// to every pure-gauge direction and the connection block of the Jacobian is
/// singular along all of them — one null direction per vertex, less one per
/// connected component. That is a property of the theory, not a defect of the
/// discretization, and a solver that inverted the Jacobian would be inverting a
/// singular matrix. The step is therefore the minimum-norm least-squares
/// solution from a complete orthogonal decomposition, which is the unique
/// solution orthogonal to the gauge orbit: the connection moves only in
/// physical directions and the gauge is left where it was.
class HolomorphicRelaxation {
 public:
  /// Build a solve over an action.
  ///
  /// @param action The action whose stationarity equations are solved. The
  ///   instance is held by value and its multipliers are advanced by the solve;
  ///   the complex it refers to is the object whose edge lengths and phases the
  ///   solve writes.
  /// @param declaration The numerical controls of the root find.
  /// @throws std::invalid_argument when the declared contour node count is
  ///   below five, when the contour radius is not positive, or when no field is
  ///   declared relaxable.
  HolomorphicRelaxation(JointAction action,
                        HolomorphicRelaxationDeclaration declaration);

  /// Run the solve, writing the relaxed fields into the complex as it goes.
  [[nodiscard]] HolomorphicRelaxationReport solve();

  /// The action, carrying the multipliers as the solve left them.
  [[nodiscard]] const JointAction &action() const noexcept { return action_; }

  /// The Jacobian of the stationarity system at the current point, flat
  /// row-major, with the rows in the residual's block order and the columns in
  /// the variable order — the relaxed length coordinates, then the relaxed link
  /// coordinates, then the relaxed multipliers. Exposed so that the derivative
  /// the solve steps along can be inspected and checked against an independent
  /// evaluation rather than only trusted.
  ///
  /// Forming it moves each coordinate around its own contour in turn and
  /// restores the complex exactly after every evaluation, so the geometry is
  /// the same before and after the call.
  [[nodiscard]] std::vector<std::complex<double>> jacobian() const;

  /// The number of rows of `jacobian`, which is the number of equations in
  /// scope.
  [[nodiscard]] std::size_t equationCount() const;

  /// The number of columns of `jacobian`, which is the number of variables in
  /// scope.
  [[nodiscard]] std::size_t variableCount() const;

 private:
  JointAction action_;
  HolomorphicRelaxationDeclaration declaration_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_HOLOMORPHICRELAXATION_H
