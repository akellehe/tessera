// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_HOLOMORPHICRELAXATION_H
#define TESSERA_COBORDISM_HOLOMORPHICRELAXATION_H

#include <array>
#include <complex>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cobordism/JointAction.h"

namespace tessera::cobordism {

/// # RebuiltBand
///
/// One band of the carrier's spectrum as a `CovarianceRebuild` read it at a
/// point: which modes of the eigendecomposition it holds, and how the rebuilt
/// state is composed of its Riesz projector
/// \f$ P_b=\sum_{k\in b}v_kw_k^{\mathsf T} \f$.
struct RebuiltBand {
  /// The indices of the band's modes: columns of
  /// `RebuiltCarrierState::eigenvectors` and rows of `leftEigenvectors`.
  std::vector<std::size_t> modes;
  /// \f$ n_b/r_b \f$, the weight of \f$ P_b \f$ in the covariance
  /// \f$ \Gamma=\sum_b(n_b/r_b)P_b \f$.
  double covarianceWeight = 0.0;
  /// Whether \f$ P_b \f$ is a summand of the constrained fiber's projector
  /// (`RebuiltCarrierState::momentProjector`).
  bool inFiber = false;
};

/// # RebuiltCarrierState
///
/// What a `CovarianceRebuild` sets on the action at one point: the carried
/// covariance \f$ \Gamma \f$ and, when the system imposes spectral constraints
/// on a fiber, the fiber's Riesz projector
/// (`JointActionDeclaration::momentProjector`) and the projectors of the
/// pinned bands, all flat row-major over the carrier's cells; and what the
/// analytic Jacobian of the self-consistent system needs beside them. The
/// derivative of the rebuilt state along a coordinate is the first-order
/// perturbation of each band's Riesz projector
/// (`rieszProjectorDerivative`), which is formed from the eigendecomposition
/// of the operator the bands were read on and from the bands' composition. A
/// rebuild that leaves `bands` empty supplies a state whose derivative
/// cannot be formed, and `HolomorphicRelaxation::jacobian` refuses by name.
struct RebuiltCarrierState {
  std::vector<std::complex<double>> covariance;
  /// Empty when the system imposes no constraint on a fiber.
  std::vector<std::complex<double>> momentProjector;
  /// The band projectors of the system's band-mean constraints
  /// (`JointActionDeclaration::momentBandProjectors`), in the constraints'
  /// order; empty when it imposes none.
  std::vector<std::vector<std::complex<double>>> bandProjectors;
  /// The eigenvalues of the operator the bands were read on, in the
  /// eigensolver's order.
  std::vector<std::complex<double>> eigenvalues;
  /// \f$ V \f$, the right eigenvectors as columns, flat row-major.
  std::vector<std::complex<double>> eigenvectors;
  /// \f$ W=V^{-1} \f$, whose rows are the left eigenvectors, flat row-major.
  std::vector<std::complex<double>> leftEigenvectors;
  /// The bands the covariance fills.
  std::vector<RebuiltBand> bands;
  /// For each entry of `bandProjectors`, the index in `bands` of the band it
  /// is the projector of.
  std::vector<std::size_t> bandProjectorBands;
  /// The operators \f$ D(g) \f$ of a declared band symmetry, each flat
  /// row-major, when the bands were read on the group average
  /// \f$ |G|^{-1}\sum_gD(g)^{-1}hD(g) \f$ of the carrier rather than on the
  /// carrier itself; empty otherwise. The variation of the averaged operator
  /// is the same average of the carrier's variation.
  std::vector<std::vector<std::complex<double>>> bandSymmetry;
};

/// The first-order variation of the Riesz projector of an isolated band under
/// a variation \f$ \delta h \f$ of the operator, from the eigendecomposition
/// \f$ h=V\Lambda V^{-1} \f$ at the point: with \f$ v_k \f$ the columns of
/// \f$ V \f$ and \f$ w_k^{\mathsf T} \f$ the rows of \f$ W=V^{-1} \f$,
/// \f[
///   \delta P=\sum_{k\in b}\sum_{j\notin b}
///     \frac{v_kw_k^{\mathsf T}\,\delta h\,v_jw_j^{\mathsf T}
///          +v_jw_j^{\mathsf T}\,\delta h\,v_kw_k^{\mathsf T}}{\lambda_k-\lambda_j},
/// \f]
/// the residue of \f$ (\zeta-h)^{-1}\,\delta h\,(\zeta-h)^{-1} \f$ on a
/// contour around the band's eigenvalues. It is exact for a band whose
/// eigenvalues are separated from the rest of the spectrum, whatever their
/// degeneracy inside the band: the pairs inside the band contribute no
/// residue. Every matrix is flat row-major.
/// @param eigenvalues \f$ \lambda \f$, one per mode.
/// @param eigenvectors \f$ V \f$.
/// @param leftEigenvectors \f$ W \f$.
/// @param modes The band's modes, as indices.
/// @param operatorVariation \f$ \delta h \f$.
/// @throws std::invalid_argument when the sizes disagree, when a mode index
///   is out of range, or when an eigenvalue outside the band equals one
///   inside it exactly, so that the quotient has no value. Eigenvalues that
///   are close and not equal are divided by their difference as they are.
[[nodiscard]] std::vector<std::complex<double>> rieszProjectorDerivative(
    const std::vector<std::complex<double>> &eigenvalues,
    const std::vector<std::complex<double>> &eigenvectors,
    const std::vector<std::complex<double>> &leftEigenvectors,
    const std::vector<std::size_t> &modes,
    const std::vector<std::complex<double>> &operatorVariation);

/// # CovarianceRebuild
///
/// The rule by which the self-consistent system rebuilds the carried
/// covariance \f$ \Gamma \f$ (and the constrained fiber's projector, when
/// one is declared) from the carrier operator at the point it is evaluated
/// at. With it `HolomorphicRelaxation` poses the self-consistent system
/// \f$ F_{\rm sc}(z,U)=F(z,U,\Gamma(z,U))=0 \f$ instead of the stationarity
/// at a fixed \f$ \Gamma \f$: the residual and every Jacobian column read
/// \f$ \Gamma \f$ rebuilt at the point. The equations are the same; only
/// \f$ \Gamma \f$ is no longer held.
struct CovarianceRebuild {
  /// The state at the geometry the action currently refers to, which it
  /// leaves as it found it.
  std::function<RebuiltCarrierState(const JointAction &)> at;
};

/// # HeldMonopoleSector
///
/// One declared cluster whose monopole sector is boundary data of the system:
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
/// The declaration of a stationarity system: which fields are its variables,
/// which coordinates they share, which sectors are held, and the rank
/// tolerance of its linear solves. None of the fields changes the action.
struct HolomorphicRelaxationDeclaration {
  /// Whether the squared lengths \f$ z_e \f$ are variables of the system. When
  /// false they are held at their current values and the length stationarity
  /// equations are dropped from the system, because requiring an equation of a
  /// frozen variable would make the system overdetermined rather than partially
  /// relaxed.
  bool relaxLengths = true;

  /// Whether the links \f$ U_e \f$ are variables of the system, under the same
  /// rule as `relaxLengths`.
  bool relaxLinks = true;

  /// Whether the multipliers \f$ \xi_j \f$ are variables of the system. When
  /// false they are held at their declared values and the moment equations are
  /// dropped, which turns a constrained system into an unconstrained one at a
  /// fixed external source.
  bool relaxMultipliers = true;

  /// The relative threshold below which a singular value of the scaled
  /// Jacobian (`HolomorphicRelaxation::variableScales`) counts as zero in the
  /// minimum-norm solve of the linearized system: the rank is the number of
  /// singular values \f$ \sigma_i>\tau\,\sigma_{\max} \f$, decided on the
  /// singular values themselves and not on a pivoted-QR diagonal, whose
  /// magnitudes only bracket them. The rank of the held faces'
  /// coboundary, whose kernel is the link-modulus directions the held
  /// sectors leave free, is decided at the same threshold relative to that
  /// matrix's largest singular value.
  double rankTolerance = 1e-15;

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
  /// the system's variables are one squared length and one link per class,
  /// and the equation of a class is the sum of its edges' stationarity
  /// equations, which is the derivative of the whole action along the shared
  /// coordinate. This is how a \f$ k \f$-sheeted support is relaxed as one
  /// base field (WP v17 §8, "Sheet convention (adopted)": equal squared
  /// lengths and equal connection values on corresponding edges). The
  /// members of a class must carry equal fields at the point the system is
  /// posed at.
  std::vector<std::size_t> edgeClasses;

  /// With `edgeClasses`, the orientation of each edge relative to its class:
  /// \f$ +1 \f$ when the edge's stored link is the class's link, \f$ -1 \f$
  /// when it is its inverse. Empty means \f$ +1 \f$ for every edge.
  std::vector<int> edgeClassOrientations;

  /// The monopole sectors held as boundary data (controlled synthesis, WP v18
  /// Section 11.1). The monopole number through a declared cluster's bounding
  /// cut is boundary data, held on that cut by this rule; it is not an
  /// invariant of continuous relaxation, because a face flux is defined
  /// modulo \f$ 2\pi \f$ and the number changes by \f$ \pm1 \f$ whenever a
  /// face holonomy on the cut passes through \f$ -1 \f$, so an odd sector
  /// persists only where it is held or where the dynamics keeps the cut's
  /// holonomies away from \f$ -1 \f$; in the bulk, and on every grown level,
  /// it is read and never held. For every declared sector the system keeps (i)
  /// the modulus of every
  /// face holonomy on its cut, so a holonomy on the unit circle stays on it,
  /// and (ii) the monopole number: a system cannot be posed at a point where
  /// a sector reads a number other than its declared one, so a drive scores
  /// such a point as having no value. The arguments of the face holonomies
  /// and every other connection degree of freedom relax freely.
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
  /// kernel of \f$ C \f$, with its rank decided at the declared rank
  /// tolerance times that system's largest singular value
  /// (`HolomorphicNewtonStep::constrainedRank`). It is the constrained
  /// Newton (Gauss-Newton) step, and so a descent direction for the residual
  /// norm whenever it is not zero. The complex equations can outnumber the
  /// real directions the held set leaves free, so the step's own linearized
  /// residual (`HolomorphicNewtonStep::linearResidual`) need not vanish; it
  /// is reported.
  std::vector<HeldMonopoleSector> heldSectors;

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
  /// force norm a read reports.
  double gradientNorm = 0.0;
};

/// The scale of every length coordinate of a system: the modulus of the
/// squared length the coordinate carries (the largest over the edges of a
/// declared class; one when every one of them is zero), in the order of the
/// length block of `HolomorphicRelaxation::jacobian`. A length equation
/// \f$ \partial S/\partial z \f$ times its scale is dimensionless, as the
/// link equations are.
[[nodiscard]] std::vector<double> lengthCoordinateScales(
    const JointAction &action,
    const HolomorphicRelaxationDeclaration &declaration);

/// # HolomorphicNewtonStep
///
/// The step that solves the stationarity system to first order about a
/// point, \f$ Jd=-R \f$ with \f$ R \f$ the residual and \f$ J \f$ its
/// closed-form Jacobian there (`HolomorphicRelaxation::newtonStep`), with the
/// rank decisions it was solved at.
struct HolomorphicNewtonStep {
  /// \f$ d \f$, one entry per variable in the order of
  /// `HolomorphicRelaxation::jacobian`'s columns: the step of each relaxed
  /// squared length, the Maurer-Cartan increment \f$ \delta \f$ of each
  /// relaxed link (\f$ U\mapsto Ue^{\delta} \f$ on the coordinate's
  /// orientation), and the step of each relaxed multiplier.
  std::vector<std::complex<double>> step;
  /// \f$ \lVert R\rVert_2 \f$ at the point.
  double residualNorm = 0.0;
  /// The numerical rank of the scaled Jacobian \f$ DJD \f$
  /// (`HolomorphicRelaxation::variableScales`): the number of its singular
  /// values above `rankTolerance` times the largest. The singular values
  /// reported below are those of \f$ DJD \f$.
  std::size_t jacobianRank = 0;
  /// The declared relative threshold of the rank decisions
  /// (`HolomorphicRelaxationDeclaration::rankTolerance`).
  double rankTolerance = 0.0;
  double largestSingularValue = 0.0;
  /// The smallest singular value counted as nonzero; not a number at rank
  /// zero.
  double smallestRetainedSingularValue = 0.0;
  /// The largest singular value counted as zero; zero at full rank.
  double largestDiscardedSingularValue = 0.0;
  /// The ratio of the two, infinite at full rank.
  double rankGap = 0.0;
  /// Whether the step was solved over the steps that keep the held sectors'
  /// face-holonomy moduli (`HolomorphicRelaxationDeclaration::heldSectors`).
  bool constrained = false;
  /// With `constrained`, the rank of the real system the step was solved
  /// from: the number of its singular values above `rankTolerance` times its
  /// own largest singular value.
  std::size_t constrainedRank = 0;
  /// With `constrained`, the ratio of the smallest singular value counted as
  /// nonzero to the largest counted as zero in that system; infinite at full
  /// rank; not a number otherwise.
  double constrainedRankGap = 0.0;
  /// \f$ \lVert Jd+R\rVert_2/\lVert R\rVert_2 \f$: what the step leaves of
  /// the linearized equations. It is at rounding for the minimum-norm step
  /// when \f$ R \f$ lies in the range of \f$ J \f$, and need not vanish for
  /// a constrained step, whose real unknowns can be fewer than the equations.
  double linearResidual = 0.0;
};

/// # HolomorphicLinearization
///
/// The stationarity system linearized at one point
/// (`HolomorphicRelaxation::linearization`): the singular value decomposition
/// of its closed-form Jacobian \f$ J \f$ there and, with held sectors, that
/// of the real system over the steps that keep the held moduli. It is kept
/// so that \f$ Jd=b \f$ is solved in the same sense for every right-hand
/// side: the Newton step is the solution for \f$ b=-R \f$, and the step of a
/// higher order solves one such system per order with the one Jacobian.
class HolomorphicLinearization {
 public:
  struct Implementation;
  explicit HolomorphicLinearization(
      std::shared_ptr<const Implementation> implementation);

  /// The minimum-norm least-squares solution \f$ d \f$ of \f$ Jd=b \f$, the
  /// singular values at or below the declared rank tolerance times the
  /// largest counted as zero; with held sectors, the minimum-norm
  /// least-squares solution over the steps that keep every held
  /// face-holonomy modulus, its rank decided the same way on that system's
  /// own singular values. \p rightHandSide has one entry per equation, in
  /// the residual's block order.
  /// @throws std::invalid_argument when \p rightHandSide has another size.
  [[nodiscard]] std::vector<std::complex<double>> solve(
      const std::vector<std::complex<double>> &rightHandSide) const;

  /// The solution for \f$ b=-R \f$ with the rank decisions of the
  /// linearization.
  [[nodiscard]] const HolomorphicNewtonStep &newtonStep() const noexcept;

  /// \f$ R \f$ at the point, any added residual included.
  [[nodiscard]] const std::vector<std::complex<double>> &residual()
      const noexcept;

 private:
  std::shared_ptr<const Implementation> implementation_;
};

/// # HolomorphicRelaxation
///
/// The holomorphic stationarity system of `JointAction`,
/// \f[
///   \frac{\partial S}{\partial z_e}=0,\qquad
///   U_e\frac{\partial S}{\partial U_e}=0,\qquad
///   p_j(h)-p_j^{\star}=0 ,
/// \f]
/// at one point: its residual, its closed-form Jacobian and the step that
/// solves it to first order there (`residual`, `jacobian`, `linearization`,
/// `newtonStep`). It moves nothing. The search for a stationary point is a
/// drive of `MultiCobordism` with this system as its objective's scalar and
/// direction (`tessera.drivers.cell_solve`).
///
/// Reference: Kato, "Perturbation Theory for Linear Operators", Springer,
/// Chapter II, for the analytic dependence of the Riesz projector of an
/// isolated group of eigenvalues on the operator, which the Jacobian of the
/// self-consistent system rests on.
///
/// These are the complex equations themselves, not a real projection of
/// them: both the real and the imaginary part of every equation vanish at a
/// solution, because each is one complex equation.
///
/// ## The variables
///
/// The variables are the complex squared lengths \f$ z_e=\ell_e^2 \f$, the
/// Maurer-Cartan increments \f$ \delta_e \f$ of the links on each edge's
/// stored orientation, and the multipliers \f$ \xi_j \f$. A step
/// \f$ \delta_e \f$ sends \f$ U_e\mapsto U_e\,e^{\delta_e} \f$. The mesh
/// stores the connection as the phase \f$ \varphi_e \f$ of
/// \f$ U_e=e^{i\varphi_e} \f$, and the identical update on that storage is
/// \f$ \varphi_e\mapsto\varphi_e-i\,\delta_e \f$, an exact rewriting of the
/// multiplicative step that forms no logarithm of a link. With
/// `HolomorphicRelaxationDeclaration::edgeClasses` the variables are shared
/// coordinates, each carried by every edge of its class, and each equation
/// is the sum of the stationarity equations of its class's edges.
///
/// ## Why the linearized system is solved in the minimum-norm sense
///
/// The action is gauge invariant, so its link stationarity vector is
/// orthogonal to every pure-gauge direction and the connection block of the
/// Jacobian is singular along all of them, one null direction per vertex
/// less one per connected component. That is a property of the theory. The
/// step is therefore the minimum-norm least-squares solution from the
/// singular value decomposition, with the singular values at or below
/// `rankTolerance` times the largest counted as zero, which is the solution
/// orthogonal to the numerical null space: the connection moves only in
/// physical directions and the gauge is left where it was. The rank and the
/// gap at the rank decision are reported with the step. With held sectors
/// the same rule is applied to the real least-squares system over the steps
/// that keep the held moduli
/// (`HolomorphicRelaxationDeclaration::heldSectors`).
///
/// ## The units of the linearized system
///
/// The Jacobian's blocks carry different units: a length row is
/// \f$ \partial S/\partial z \f$ and a length column multiplies a step of
/// \f$ z \f$, while the link and multiplier rows and columns are
/// dimensionless. On a cell whose squared lengths are of order
/// \f$ 10^{9} \f$ the length blocks are \f$ 10^{-9} \f$ and
/// \f$ 10^{-18} \f$ of the link block, and a rank decision on \f$ J \f$
/// itself would count every singular value that involves a length as zero.
/// The equation \f$ J\,d=-R \f$ is therefore solved as
/// \f$ (DJD)\,y=-DR \f$, \f$ d=Dy \f$, with \f$ D \f$ the diagonal of
/// `variableScales`: the modulus of the squared length for a length
/// coordinate, one otherwise. Its length rows are \f$ |z|\,\partial S/
/// \partial z \f$ and its length unknowns are \f$ dz/|z| \f$. It is the
/// same equation, so where \f$ J \f$ has full rank the step is the same;
/// the rank decision and the minimum norm are those of the scaled equation.
///
/// ## The Jacobian
///
/// The Jacobian of the stationarity system is assembled analytically
/// (`jacobian`). Its geometric blocks are the Hessian of the action at fixed
/// carried state, `JointAction::actionHessian`: the Regge Hessian on the
/// declared sheets, the Villain Hessian in the Maurer-Cartan increments, and
/// the contraction of the covariant operator's second derivatives against
/// \f$ w_M\Gamma+\sum_j\xi_jX_j \f$ together with the variation of the
/// power sums' matrices \f$ X_j \f$ through the operator. The multiplier
/// columns are the constraints' gradients `JointAction::momentGradient`, and
/// the constraint rows are the same gradients: the derivative of a
/// constraint's value at a spectral projector, whose own variation is
/// off-diagonal between its range and its kernel and contributes no trace.
/// Under declared edge classes every entry is the double sum over the two
/// classes' members, each link member on its orientation, since the equation
/// of a class is the sum of its edges' equations and its coordinate moves
/// every edge of the class.
///
/// ## The self-consistent system
///
/// Built with a `CovarianceRebuild`, the system rebuilds \f$ \Gamma \f$ from
/// the carrier operator at the point, so it is the joint system
/// \f$ F_{\rm sc}(z,U)=0 \f$ of `SelfConsistentMeanField` with the Jacobian
/// \f$ H_{\rm sc} \f$ of the self-consistent force. The equations are those
/// of the joint action with \f$ \Gamma \f$ no longer held. The Jacobian then
/// carries, on
/// every geometric column, the response of the equations to the state's
/// variation along that coordinate: with \f$ \delta h \f$ the operator's
/// derivative along the coordinate, the variation of each band's Riesz
/// projector is `rieszProjectorDerivative`, that of the covariance
/// \f$ \sum_b(n_b/r_b)\,\delta P_b \f$ and that of the pinned fiber's
/// projector the sum over its bands, and the response is
/// `JointAction::stationarityStateVariation`. The perturbation is exact for
/// an isolated band; a band close to another eigenvalue is perturbed by the
/// same formula, whose quotients are then large, and the isolation of every
/// band is reported with the bands (`BandRead::bandIsolation`).
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
  /// Pose the system of an action at the action's current point.
  ///
  /// @param action The action whose stationarity equations are posed. The
  ///   instance is held by value; the complex it refers to is read and not
  ///   written.
  /// @param declaration The variables, their shared coordinates, the held
  ///   sectors and the rank tolerance.
  /// @param rebuild Empty (the default) to hold the action's covariance fixed;
  ///   otherwise the rule that rebuilds \f$ \Gamma \f$ at the point.
  /// @throws std::invalid_argument when no field is declared relaxable, when
  ///   declared edge classes do not cover the edges, skip a class index,
  ///   carry an orientation other than plus or minus one, or carry unequal
  ///   fields inside a class, or when a held sector reads a monopole number
  ///   other than its declared one at the point.
  HolomorphicRelaxation(JointAction action,
                        HolomorphicRelaxationDeclaration declaration,
                        CovarianceRebuild rebuild = {});

  /// The action the system is posed on, with its multipliers.
  [[nodiscard]] const JointAction &action() const noexcept { return action_; }

  /// The analytic Jacobian of the stationarity system at the current point,
  /// flat row-major, with the rows in the residual's block order and the
  /// columns in the variable order — the relaxed length coordinates, then the
  /// relaxed link coordinates, then the relaxed multipliers. Under declared
  /// edge classes a length or link coordinate is one class, in class order.
  /// Exposed so that the derivative a step is taken along can be inspected
  /// and checked against an independent evaluation rather than only trusted.
  /// With a `CovarianceRebuild` it is the Jacobian of the self-consistent
  /// force, the state rebuilt at the point and its derivative taken from the
  /// perturbation of the bands' Riesz projectors. The geometry is read and
  /// not written.
  /// @throws std::invalid_argument when a rebuild supplies no band data, or
  ///   when an eigenvalue inside a rebuilt band equals one outside it
  ///   exactly, where the perturbation of its projector has no value.
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

  /// The step that solves the system to first order about the current point:
  /// the minimum-norm least-squares solution \f$ d \f$ of \f$ Jd=-R \f$ from
  /// the singular value decomposition of the closed-form Jacobian, the
  /// singular values at or below the declared rank tolerance times the
  /// largest counted as zero. The Jacobian is singular along every pure-gauge
  /// direction, because the action is gauge invariant, and the minimum-norm
  /// solution is the one orthogonal to that null space. With held sectors the
  /// step is the minimum-norm least-squares solution over the steps that keep
  /// every held face-holonomy modulus (the real part of the link block in the
  /// kernel of the held faces' coboundary), from the singular value
  /// decomposition of the real system in the real unknowns that parametrize
  /// those steps, its rank decided at the declared rank tolerance times that
  /// system's own largest singular value. The geometry is read and not
  /// written.
  /// @throws std::invalid_argument, std::domain_error or std::runtime_error
  ///   when the residual or the Jacobian has no value at the point.
  [[nodiscard]] HolomorphicNewtonStep newtonStep() const;

  /// The scale of every variable, in the order of `jacobian`'s columns: for
  /// a relaxed squared length, the modulus of the squared length its
  /// coordinate carries (the largest over the edges of a declared class; one
  /// when every one of them is zero); one for a link's increment and for a
  /// multiplier. The linearized system is decomposed with its rows and
  /// columns multiplied by these (see the class documentation).
  [[nodiscard]] std::vector<double> variableScales() const;

  /// The system linearized at the current point, for solves against several
  /// right-hand sides (`HolomorphicLinearization`). \p addedResidual and
  /// \p addedJacobian, each empty or of the system's size (the Jacobian flat
  /// row-major), are added to \f$ R \f$ and to \f$ J \f$ first: the gradient
  /// and the Hessian, on the system's variables, of a term the caller adds to
  /// the action.
  /// @throws std::invalid_argument when an added block has another size, and
  ///   as `newtonStep` does.
  [[nodiscard]] HolomorphicLinearization linearization(
      const std::vector<std::complex<double>> &addedResidual = {},
      const std::vector<std::complex<double>> &addedJacobian = {}) const;

  /// The monopole number of every held sector at the current point, in the
  /// declaration's order: the sum of the principal arguments of the sector's
  /// outward face holonomies over \f$ 2\pi \f$. Empty when no sector is held.
  [[nodiscard]] std::vector<int> sectorMonopoleNumbers() const;

  /// The logarithm of the modulus of every held face holonomy at the current
  /// point, sector by sector in the declaration's order. Empty when no
  /// sector is held.
  [[nodiscard]] std::vector<double> heldLogModuli() const;

 private:
  JointAction action_;
  HolomorphicRelaxationDeclaration declaration_;
  CovarianceRebuild rebuild_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_HOLOMORPHICRELAXATION_H
