// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_JOINTACTION_H
#define TESSERA_COBORDISM_JOINTACTION_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cobordism/HodgeLaplacian.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # SpectralMomentConstraint
///
/// One holomorphic spectral constraint of the targeted action
/// \f$ S_{\rm spec} \f$ of Section 3 of the whitepaper: the power-sum
/// invariant \f$ p_j(h)=\operatorname{tr}(h^j) \f$ of the complex edge-mode
/// operator \f$ h \f$ is required to equal a prescribed complex number
/// \f$ p_j^{\star} \f$, and the requirement is imposed by an independent
/// complex Lagrange multiplier \f$ \xi_j \f$ rather than by a penalty.
///
/// The multiplier is a solved-for variable, not a configured weight. The
/// constraint contributes the term \f$ \xi_j\,(p_j(h)-p_j^{\star}) \f$ to the
/// action, whose stationarity in \f$ \xi_j \f$ is the full complex equation
/// \f$ p_j(h)-p_j^{\star}=0 \f$ — both the real and the imaginary part, with no
/// residual norm minimized in place of the equation. `multiplier` holds the
/// current value of \f$ \xi_j \f$, which a solver updates alongside the
/// geometry; its initial value is a starting point for the root find and
/// nothing else.
///
/// The whitepaper permits these target terms only in explicitly labelled
/// controlled synthesis. They are absent in emergence mode, where every
/// particle-specific observable is read after the stationary solve instead of
/// being inserted as a target.
struct SpectralMomentConstraint {
  /// The moment index \f$ j \ge 1 \f$. It is the power the operator is raised
  /// to in \f$ p_j(h)=\operatorname{tr}(h^j) \f$ and is not a simplicial
  /// degree.
  int order = 1;
  /// The target \f$ p_j^{\star}=\sum_a (\lambda_a^{\star})^j \f$, the same
  /// power sum of the prescribed eigenvalue multiset.
  std::complex<double> target{0.0, 0.0};
  /// The complex Lagrange multiplier \f$ \xi_j \f$ at the current point of a
  /// solve.
  std::complex<double> multiplier{0.0, 0.0};
};

/// # JointActionDeclaration
///
/// Everything that fixes which action \f$ S(z,U,\Gamma) \f$ a `JointAction`
/// instance is. Plain data, so an instance can be built, copied and recorded
/// without reaching any engine state.
struct JointActionDeclaration {
  /// The simplicial degree \f$ k \f$ of the one-particle carrier. The complex
  /// edge-mode operator of the action is \f$ h = h_k(z,U) \f$, the metric Hodge
  /// operator at that degree, and the covariance \f$ \Gamma \f$ is a matrix over
  /// the \f$ k \f$-cells of the complex.
  int carrierDegree = 1;

  /// \f$ w_R \f$, the coefficient multiplying the dual Lorentzian Regge action
  /// \f$ S_{\rm Regge}(z)=\sum_h|\!\star\!h|\,\varepsilon_h \f$.
  ///
  /// The whitepaper's Section 7 identification is \f$ w_R = 1/(8\pi G) \f$ in
  /// lattice units, so a caller working in terms of the backreaction coupling
  /// \f$ \kappa = 8\pi G \f$ sets this to \f$ 1/\kappa \f$. The coefficient is
  /// declared here rather than derived, because the units in which the carried
  /// state's bilinear density is measured are the caller's to fix.
  double gravitationalWeight = 1.0;

  /// \f$ w_H \f$, the coefficient multiplying the face-holonomy term
  /// \f$ S_{\rm hol}(U) \f$. It is the parameter the whitepaper calls
  /// \f$ \beta \f$ when it takes the bare connection stiffness in
  /// Wilson-plaquette form.
  double holonomyWeight = 0.0;

  /// \f$ w_M \f$, the coefficient multiplying the matter term
  /// \f$ S_{\rm matter}=\operatorname{tr}(\Gamma\,h(z,U)) \f$, the carried
  /// state's bilinear action density. Zero leaves the geometry blind to the
  /// carried state, which is the whitepaper's strict-emergence mode.
  double matterWeight = 0.0;

  /// The carried covariance \f$ \Gamma=\Phi\tilde\Phi^{\mathsf T} \f$, flat
  /// row-major over the \f$ k \f$-cells of the complex in the canonical
  /// `ChainComplex::kSimplexVertices` order, with \f$ k \f$ = `carrierDegree`.
  ///
  /// Complex bilinear throughout: no adjoint is taken of it, it is not required
  /// to be Hermitian, and the matter term is the plain complex trace
  /// \f$ \operatorname{tr}(\Gamma h) \f$ rather than the real part of one. An
  /// empty vector means no carried state, and then the matter term is exactly
  /// zero whatever `matterWeight` says.
  std::vector<std::complex<double>> covariance;

  /// The declared holomorphic spectral constraints. Empty in emergence mode.
  std::vector<SpectralMomentConstraint> momentConstraints;

  /// Where the carrier operator's metric comes from.
  ///
  /// `WhitneyPencil` is the whitepaper's \f$ W_k = M_k^{-1} \f$ and gives the
  /// covariant \f$ h_k(z,U) \f$, which depends on the connection. Under
  /// `DiagonalWeights` the operator is blind to \f$ U \f$ at every degree, so
  /// the matter and spectral terms then contribute nothing to the link
  /// stationarity equation and the only connection dependence left in the
  /// action is the face-holonomy term.
  HodgeLaplacian::MetricSource metricSource =
      HodgeLaplacian::MetricSource::WhitneyPencil;
};

/// # JointAction
///
/// The gauge-invariant joint action \f$ S(z,U,\Gamma) \f$ of Section 3 and
/// Section 13 of the whitepaper, and its exact holomorphic stationarity
/// equations.
///
/// Reference: Regge, "General relativity without coordinates", Nuovo Cimento
/// 19, 558 (1961).
/// Reference: Wilson, "Confinement of quarks", Physical Review D 10, 2445
/// (1974), for the plaquette form of the face-holonomy term.
///
/// The action is the sum of four terms,
/// \f[
///   S(z,U,\Gamma) = w_R\,S_{\rm Regge}(z) + w_H\,S_{\rm hol}(U)
///                 + w_M\,S_{\rm matter}(z,U,\Gamma) + S_{\rm spec}(z,U;\xi),
/// \f]
/// in the two edge fields of the microscopic state — the complex squared length
/// \f$ z_e=\ell_e^2 \f$ and the multiplicative connection
/// \f$ U_e \in \mathbb{C}^{*} \f$ — and the carried covariance \f$ \Gamma \f$:
///
/// * \f$ S_{\rm Regge}(z)=\sum_h |\!\star\! h|\,\varepsilon_h \f$ is the dual
///   Lorentzian Regge action (`simulations::ReggeSolver::dualReggeAction`),
///   a holomorphic function of the squared lengths and independent of
///   \f$ U \f$.
/// * \f$ S_{\rm hol}(U)=\sum_\tau\bigl(1-\tfrac12(\mathcal F_\tau
///   +\mathcal F_\tau^{-1})\bigr) \f$ runs over the triangles \f$ \tau \f$ of
///   the complex, with the branch-free face holonomy
///   \f$ \mathcal F_\tau=\prod_{e\subset\partial\tau}U_e^{\epsilon_{\tau e}} \f$
///   and \f$ \epsilon_{\tau e} \f$ the integer incidence of \f$ \partial_2 \f$.
///   It is a holomorphic function on \f$ (\mathbb{C}^{*})^{|E|} \f$: both
///   \f$ \mathcal F_\tau \f$ and \f$ \mathcal F_\tau^{-1} \f$ are Laurent
///   monomials in the links, so no argument, logarithm or modulus of a link is
///   ever taken. On a unimodular connection, writing
///   \f$ \mathcal F_\tau=e^{i\Theta_\tau} \f$, it is the real Wilson plaquette
///   sum \f$ \sum_\tau(1-\cos\Theta_\tau) \f$, and its second variation in the
///   real angles at trivial holonomy is the up-Laplacian
///   \f$ L_1^{\rm up}=\partial_2\partial_2^{\mathsf T} \f$, which vanishes on
///   pure-gauge directions and is \f$ 4 \f$ on the coexact block of the regular
///   tetrahedron. The symmetric combination
///   \f$ \tfrac12(\mathcal F+\mathcal F^{-1}) \f$ is used rather than
///   \f$ \mathcal F \f$ alone because the latter is not stationary at trivial
///   holonomy, and rather than \f$ \operatorname{Re}\mathcal F \f$ because that
///   is not holomorphic.
/// * \f$ S_{\rm matter}(z,U,\Gamma)=\operatorname{tr}(\Gamma\,h(z,U)) \f$ is the
///   carried state's bilinear action density \f$ \tilde\psi^{\mathsf T}h\psi \f$
///   reduced on \f$ \Gamma \f$ by Wick's theorem. It is the one channel from the
///   state to the geometry, it is complex, and no adjoint of \f$ h \f$ and no
///   real projection enters it.
/// * \f$ S_{\rm spec}(z,U;\xi)=\sum_j \xi_j\,(p_j(h)-p_j^{\star}) \f$ carries the
///   declared `SpectralMomentConstraint`s, with
///   \f$ p_j(h)=\operatorname{tr}(h^j) \f$.
///
/// The carrier operator \f$ h=h_k(z,U) \f$ is `HodgeLaplacian::laplacian` at the
/// declared degree under the declared metric source. It is generally
/// non-normal, it is never symmetrized, and the power sums are invariant under
/// the similarity the gauge group acts by, so every spectral quantity built
/// here is constant along gauge orbits.
///
/// ## The stationarity equations
///
/// The action is holomorphic in every variable, so the stationarity conditions
/// are the complex equations
/// \f[
///   \frac{\partial S}{\partial z_e}=0,\qquad
///   U_e\frac{\partial S}{\partial U_e}=0,\qquad
///   \frac{\partial S}{\partial \xi_j}=p_j(h)-p_j^{\star}=0 ,
/// \f]
/// and never the minimization of a selected real projection of \f$ S \f$. The
/// connection equation is written in the Maurer-Cartan coordinate
/// \f$ U^{-1}\delta U \f$, so it asks for no argument and no logarithm of a
/// link; `HolomorphicRelaxation` updates \f$ U \f$ multiplicatively by the same
/// coordinate.
///
/// The link stationarity vector is also the Ward current of Section 13.4,
/// \f$ j_{xy}=U_{xy}\,\partial S/\partial U_{xy} \f$, and `linkStationarity`
/// and `wardCurrent` are the same numbers under the two names the whitepaper
/// gives them. Because the reversed link is the inverse,
/// \f$ U_{yx}=U_{xy}^{-1} \f$, the current is odd under reversing an edge:
/// \f$ j_{yx}=-j_{xy} \f$.
///
/// ## Orders and conventions
///
/// Every per-edge vector this class returns is indexed in
/// `Spacetime::getEdgeList()` order, the order the rest of the engine's edge
/// quantities use, and every per-cell quantity in the canonical
/// `ChainComplex::kSimplexVertices` order. The link stationarity of an edge is
/// reported on that edge's **stored** source-to-target orientation, which is
/// the orientation the current's sign convention refers to.
///
/// Derivatives are plain holomorphic derivatives. There is no Wirtinger
/// \f$ \bar z \f$ component to report, because the action does not depend on
/// \f$ \bar z \f$ or on \f$ \bar U \f$ at all.
class JointAction {
 public:
  /// Build the action over a triangulation.
  ///
  /// @param spacetime The complex carrying the two edge fields. Retained for
  ///   the instance's lifetime, and read — never written — by every query here.
  /// @param declaration Which action this is: the carrier degree, the three
  ///   coefficients, the carried covariance, the spectral constraints and the
  ///   metric source.
  /// @throws std::invalid_argument when \p spacetime is null, when the carrier
  ///   degree is negative, when a declared moment order is below one, or when
  ///   the covariance is present and is not a square matrix over the
  ///   \f$ k \f$-cells of the complex.
  JointAction(std::shared_ptr<Spacetime> spacetime,
              JointActionDeclaration declaration);

  /// The declaration this instance was built from.
  [[nodiscard]] const JointActionDeclaration &declaration() const noexcept {
    return declaration_;
  }

  /// The complex the action is defined over.
  [[nodiscard]] const std::shared_ptr<Spacetime> &spacetime() const noexcept {
    return spacetime_;
  }

  /// Replace the declared multiplier \f$ \xi_j \f$ of every constraint, in the
  /// declaration order of `JointActionDeclaration::momentConstraints`. This is
  /// how a solver advances the multipliers, which are variables of the
  /// stationarity system rather than configuration.
  /// @throws std::invalid_argument when the count differs from the number of
  ///   declared constraints.
  void setMultipliers(const std::vector<std::complex<double>> &multipliers);

  /// The current multipliers \f$ \xi_j \f$, in declaration order.
  [[nodiscard]] std::vector<std::complex<double>> multipliers() const;

  /// The carrier operator \f$ h_k(z,U) \f$, flat row-major over the
  /// \f$ k \f$-cells in the canonical order. Empty when the complex carries no
  /// cell of the declared degree.
  [[nodiscard]] std::vector<std::complex<double>> carrierOperator() const;

  /// The branch-free face holonomies \f$ \mathcal F_\tau \f$, one per triangle
  /// of the complex, in the canonical order of the degree-two cells. Each is the
  /// ordered product of the links over the incidences of \f$ \partial_2 \f$ and
  /// is computed without forming a sum of phases. Empty for a complex of
  /// dimension below two.
  [[nodiscard]] std::vector<std::complex<double>> faceHolonomies() const;

  /// The power sums \f$ p_j(h)=\operatorname{tr}(h^j) \f$ of the declared
  /// constraints, in declaration order. Computed from the operator by repeated
  /// multiplication, so a defective or non-normal \f$ h \f$ needs no
  /// eigendecomposition and no eigenvalue ordering.
  [[nodiscard]] std::vector<std::complex<double>> powerSums() const;

  /// \f$ p_j(h)-p_j^{\star} \f$ for each declared constraint, in declaration
  /// order: the exact stationarity equation in \f$ \xi_j \f$, as a complex
  /// number whose vanishing is the constraint.
  [[nodiscard]] std::vector<std::complex<double>> momentResiduals() const;

  /// \f$ w_R\,S_{\rm Regge}(z) \f$.
  [[nodiscard]] std::complex<double> reggeTerm() const;
  /// \f$ w_H\,S_{\rm hol}(U) \f$.
  [[nodiscard]] std::complex<double> holonomyTerm() const;
  /// \f$ w_M\operatorname{tr}(\Gamma h(z,U)) \f$.
  [[nodiscard]] std::complex<double> matterTerm() const;
  /// \f$ \sum_j \xi_j\,(p_j(h)-p_j^{\star}) \f$.
  [[nodiscard]] std::complex<double> spectralTerm() const;
  /// The whole action \f$ S(z,U,\Gamma) \f$, the sum of the four terms above.
  [[nodiscard]] std::complex<double> value() const;

  /// \f$ \partial S/\partial z_e \f$ for every edge, in `getEdgeList()` order.
  ///
  /// Assembled from the exact analytic gradients the framework supplies: the
  /// Regge gradient `simulations::ReggeSolver::actionGradientExact` and the
  /// operator gradient `HodgeLaplacian::laplacianGradient`. No finite difference
  /// enters it, and no imaginary part is discarded: the returned number is the
  /// full complex derivative of a holomorphic function.
  [[nodiscard]] std::vector<std::complex<double>> lengthStationarity() const;

  /// \f$ U_e\,\partial S/\partial U_e \f$ for every edge, in `getEdgeList()`
  /// order and on the edge's stored orientation.
  ///
  /// The face-holonomy part is analytic in closed form,
  /// \f$ -\tfrac12 w_H \sum_\tau \epsilon_{\tau e}
  ///     (\mathcal F_\tau-\mathcal F_\tau^{-1}) \f$. The operator part uses the
  /// identity \f$ U_e\,\partial/\partial U_e = -i\,\partial/\partial\varphi_e \f$
  /// for \f$ U_e=e^{i\varphi_e} \f$ together with the exact analytic
  /// `HodgeLaplacian::laplacianPhaseGradient`; that identity is a relation
  /// between derivatives and selects no branch of the logarithm.
  [[nodiscard]] std::vector<std::complex<double>> linkStationarity() const;

  /// The Hellmann-Feynman force \f$ \operatorname{tr}(\Gamma\,\partial h/\partial z_e) \f$
  /// of Section 7, one entry per edge in `getEdgeList()` order.
  ///
  /// This is the carried state's whole contribution to the length stationarity
  /// equation, carrying neither the coefficient \f$ w_M \f$ nor the geometric
  /// terms, so a reader can see the two sides the stationary point balances
  /// separately: at a stationary point
  /// \f$ w_R\,\partial S_{\rm Regge}/\partial z_e
  ///     + w_M\operatorname{tr}(\Gamma\,\partial h/\partial z_e)=0 \f$
  /// edge by edge.
  ///
  /// The metric Hodge operator is homogeneous of degree \f$ -1 \f$ in the
  /// squared lengths, so this force obeys the Euler identity
  /// \f$ \sum_e z_e\operatorname{tr}(\Gamma\,\partial h/\partial z_e)
  ///     = -\operatorname{tr}(\Gamma h) \f$
  /// exactly. For a covariance that projects onto occupied modes the right-hand
  /// side is minus the sum of their eigenvalues, which is the whitepaper's
  /// statement that the length-weighted force of an occupied mode is dilating.
  [[nodiscard]] std::vector<std::complex<double>> hellmannFeynmanLengthForce()
      const;

  /// The connection force \f$ \operatorname{tr}(\Gamma\,U_e\,\partial h/\partial U_e) \f$,
  /// the link counterpart of `hellmannFeynmanLengthForce`, on each edge's stored
  /// orientation. It is identically zero when the carrier operator is blind to
  /// the connection, which is the case under the `DiagonalWeights` metric
  /// source.
  [[nodiscard]] std::vector<std::complex<double>> hellmannFeynmanLinkForce()
      const;

  /// The per-cell occupations \f$ n_c=\Gamma_{cc} \f$, in the canonical
  /// \f$ k \f$-cell order: the derived readout the whitepaper names, complex in
  /// general because \f$ \Gamma \f$ is a complex bilinear covariance and only a
  /// compatible \f$ * \f$-structure would make a matrix element a probability.
  /// Empty when no covariance is declared.
  [[nodiscard]] std::vector<std::complex<double>> occupationNumbers() const;

  /// The Ward current \f$ j_{xy}=U_{xy}\,\partial S/\partial U_{xy} \f$ of
  /// Section 13.4, which is `linkStationarity` under its other name.
  [[nodiscard]] std::vector<std::complex<double>> wardCurrent() const {
    return linkStationarity();
  }

  /// The discrete divergence \f$ (\partial j)_x=\sum_e (\partial_1)_{xe}\,j_e \f$
  /// of the Ward current, one complex number per vertex in the canonical
  /// degree-zero cell order.
  ///
  /// It vanishes identically for every gauge-invariant term of the action,
  /// which is the discrete Ward identity: under the infinitesimal complex gauge
  /// variation \f$ \delta U_{xy}=(-\eta_x+\eta_y)U_{xy} \f$ the first variation
  /// of a gauge-invariant functional is \f$ \langle \partial j,\eta\rangle \f$
  /// for every vertex function \f$ \eta \f$, so invariance forces
  /// \f$ \partial j=0 \f$. The Regge and holonomy terms are gauge invariant
  /// outright. The matter and spectral terms are invariant when \f$ \Gamma \f$
  /// transforms covariantly with the operator, which is exactly what a spectral
  /// projector of \f$ h \f$ does; holding a fixed \f$ \Gamma \f$ while the
  /// connection varies breaks the identity, and the size of this divergence is
  /// then the measure of that.
  [[nodiscard]] std::vector<std::complex<double>> wardCurrentDivergence() const;

  /// The whole stationarity residual vector
  /// \f$ F=(\partial S/\partial z_e;\; U_e\partial S/\partial U_e;\;
  ///        p_j(h)-p_j^{\star}) \f$,
  /// of length \f$ 2|E|+m_c \f$ in that block order: the length equations in
  /// `getEdgeList()` order, then the link equations in the same order, then the
  /// moment equations in declaration order. This is the vector a holomorphic
  /// root find drives to zero.
  [[nodiscard]] std::vector<std::complex<double>> stationarityResidual() const;

  /// The Euclidean norm of `stationarityResidual`. It is a convergence
  /// certificate reported beside the solve and not a functional that is
  /// minimized in place of the complex equations.
  [[nodiscard]] double stationarityResidualNorm() const;

  /// \f$ \partial p_j(h)/\partial z_e \f$ and
  /// \f$ U_e\,\partial p_j(h)/\partial U_e \f$ for the \p index-th declared
  /// constraint, concatenated in the same block order as
  /// `stationarityResidual`'s first two blocks, so the vector has length
  /// \f$ 2|E| \f$.
  ///
  /// This is the exact analytic column the multiplier \f$ \xi_j \f$ contributes
  /// to the Jacobian of the stationarity system, and by the symmetry of second
  /// derivatives it is also the \f$ \xi_j \f$ row.
  /// @throws std::out_of_range when \p index names no declared constraint.
  [[nodiscard]] std::vector<std::complex<double>> momentGradient(
      std::size_t index) const;

  /// The number of edges, which is the length of every per-edge vector here.
  [[nodiscard]] std::size_t edgeCount() const;

  /// The number of declared moment constraints.
  [[nodiscard]] std::size_t constraintCount() const noexcept {
    return declaration_.momentConstraints.size();
  }

  /// The eigenvalues of the carrier operator \f$ h_k(z,U) \f$, unordered as the
  /// solver produced them. Reported for the occupation rule of
  /// `SelfConsistentMeanField` and for spectral diagnostics; nothing in the
  /// action itself needs them.
  [[nodiscard]] std::vector<std::complex<double>> carrierEigenvalues() const;

  /// The spectral (Riesz) projector of the carrier operator onto the
  /// \p occupied modes selected by \p ascendingRealPart, flat row-major over the
  /// \f$ k \f$-cells.
  ///
  /// With the eigendecomposition \f$ h = V\Lambda V^{-1} \f$ the projector is
  /// \f$ \Gamma = V\,\mathrm{diag}(\chi)\,V^{-1} \f$ with \f$ \chi_a=1 \f$ on the
  /// selected modes and \f$ 0 \f$ elsewhere. Writing \f$ \Phi \f$ for the
  /// selected columns of \f$ V \f$ and \f$ \tilde\Phi^{\mathsf T} \f$ for the
  /// corresponding rows of \f$ V^{-1} \f$, this is exactly the whitepaper's
  /// \f$ \Gamma=\Phi\tilde\Phi^{\mathsf T} \f$: a matched left/right frame pair
  /// with \f$ \Gamma^2=\Gamma \f$ by construction and no adjoint anywhere. It is
  /// Hermitian only when \f$ h \f$ is normal, and it is not required to be.
  ///
  /// @param occupied How many modes are filled.
  /// @param ascendingRealPart Whether the modes are ordered by ascending real
  ///   part of the eigenvalue — the lowest-energy modes — or by ascending
  ///   modulus. The whitepaper names the filled modes "the occupied modes" and
  ///   fixes no order for a genuinely complex spectrum, so the order is declared
  ///   by the caller rather than assumed here.
  /// @throws std::invalid_argument when \p occupied exceeds the number of
  ///   \f$ k \f$-cells, or when the eigendecomposition has no inverse (a
  ///   defective operator, for which no spectral projector onto a splitting of
  ///   the spectrum is available from an eigenbasis).
  [[nodiscard]] std::vector<std::complex<double>> occupationProjector(
      std::size_t occupied, bool ascendingRealPart = true) const;

  /// The eigenvalues of the carrier operator in the order
  /// `occupationProjector` fills them, so the first \p occupied entries are the
  /// occupied ones.
  [[nodiscard]] std::vector<std::complex<double>> orderedCarrierEigenvalues(
      bool ascendingRealPart = true) const;

  /// The names of the four declared terms, in the order `value` sums them:
  /// `"regge"`, `"holonomy"`, `"matter"`, `"spectral"`.
  [[nodiscard]] static std::vector<std::string> termNames();

 private:
  std::shared_ptr<Spacetime> spacetime_;
  JointActionDeclaration declaration_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_JOINTACTION_H
