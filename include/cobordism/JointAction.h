// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_JOINTACTION_H
#define TESSERA_COBORDISM_JOINTACTION_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cobordism/HodgeLaplacian.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # OccupationOrder
///
/// Which modes of the carrier operator a quasi-free state occupies.
///
/// The whitepaper calls the filled modes "the occupied modes" and fixes no
/// order for a genuinely complex spectrum, where "lowest" is not defined by the
/// eigenvalues alone. The rule is therefore declared by the caller and recorded
/// with the run rather than assumed.
///
/// * `AscendingRealPart` — the modes of smallest real part, which is the
///   lowest-energy reading and reduces to the usual one on a Hermitian
///   specialization with real spectrum.
/// * `AscendingModulus` — the modes of smallest modulus, which is the reading
///   that orders by distance from the origin of the complex plane and is the
///   one a resolvent contour around zero selects.
enum class OccupationOrder { AscendingRealPart, AscendingModulus };

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

/// # ReggeForm
///
/// Which discretization of the Einstein-Hilbert action the Regge term of
/// `JointAction` is.
///
/// * `Primal` — Regge's own sum \f$ \sum_h |h|\,\varepsilon_h \f$ over the
///   hinges \f$ h \f$, with \f$ |h| \f$ the \f$ (d-2) \f$-content of the hinge
///   (`simulations::ReggeSolver::hingeContent`, the length of an edge on a
///   three-dimensional complex) and \f$ \varepsilon_h \f$ its complex deficit
///   angle. In three dimensions it is \f$ \sum_e l_e\,\varepsilon_e \f$, the
///   form the whitepaper's Section 7 computation takes the second variation of.
/// * `Dual` — the dual (Sorkin) form \f$ \sum_h |\!\star\! h|\,\varepsilon_h \f$,
///   with the circumcentric dual volume of each hinge in place of its content
///   (`simulations::ReggeSolver::dualReggeAction`).
///
/// The two are different functionals with different stationary points; the
/// choice is declared with every action rather than implied.
enum class ReggeForm { Primal, Dual };

/// # ReggeHinges
///
/// Which hinges the primal Regge sum runs over.
///
/// * `Interior` — only the hinges whose link closes: a hinge every one of
///   whose \f$ (d-1) \f$-faces is shared by exactly two top cells. Deficit
///   angles are defined at interior hinges; this is the sum Regge wrote and the
///   one the whitepaper's Section 7 computation evaluates ("the Regge Hessian
///   ... on its interior hinges"). On a complex with no interior hinge the
///   primal term is identically zero.
/// * `All` — every hinge that is a face of some top cell, with the library's
///   deficit \f$ 2\pi-\sum\theta \f$ at boundary hinges as well; this is the sum
///   `simulations::ReggeSolver::reggeAction` evaluates.
///
/// The rule governs the primal form only; the dual form keeps the hinge set of
/// `simulations::ReggeSolver::dualReggeAction`.
enum class ReggeHinges { Interior, All };

/// # ReggeBranch
///
/// Which Riemann sheet the primal Regge term and its derivatives are read on.
///
/// A dihedral angle is \f$ \arccos(-C_{ij}/(\sqrt{C_{ii}}\sqrt{C_{jj}})) \f$,
/// with two cofactor roots and an inverse cosine, each branched. For a
/// Euclidean tetrahedron both face cofactors \f$ C_{ii}, C_{jj} \f$ are negative
/// real, which is the cut of the principal square root: a squared length with an
/// imaginary part of \f$ 10^{-9} \f$ of either sign then puts each root on
/// either side of the cut, the principal product flips sign, and the angle jumps
/// from \f$ \theta \f$ to \f$ \pi-\theta \f$. The whitepaper's branch
/// ledger (v17: "complex Regge roots carry a Riemann-sheet label and monodromy")
/// and specification §4.2 ("for complex data the branch is fixed by
/// continuation from a Euclidean reference") exclude that.
///
/// * `Continued` — every cofactor root, inverse cosine and hinge-content root
///   carries a sheet continued from a Euclidean reference: the roots are
///   declared on their principal sheets at the real projection
///   \f$ \operatorname{Re} z^{(0)} \f$ of the declared starting geometry
///   \f$ z^{(0)} \f$ (`JointActionDeclaration::reggeStartSquaredLengths`), with
///   a real cosine pinned to the \f$ +0 \f$ side of its cuts, and continued
///   along the straight segment to \f$ z^{(0)} \f$ and then along the straight
///   segment from \f$ z^{(0)} \f$ to the geometry the mesh holds (`SheetedSqrt`,
///   `SheetedAcos`). The term is then a single-valued holomorphic function of the
///   squared lengths on every star-shaped neighbourhood of the starting geometry
///   that avoids the branch points, and it equals the principal value on real
///   input whose segment from the reference is trivial.
/// * `Principal` — every root and inverse cosine principal, a real ratio pinned
///   to the \f$ +0 \f$ side: the sheet-blind `mesh::Simplex::deficitAngle`.
///   Kept for comparison; it is discontinuous across the cuts.
///
/// The rule governs the primal form only; the dual form keeps the principal
/// branch of `simulations::ReggeSolver::dualReggeAction`.
enum class ReggeBranch { Continued, Principal };

/// # HolonomyForm
///
/// Which function of the face holonomies the face-holonomy term
/// \f$ S_{\rm hol}(U) \f$ of the joint action is.
///
/// * `Villain` — the Villain (heat-kernel) action in its character form,
///   \f$ S_{\rm hol}=-\beta_V\sum_\tau\log W_\beta(\mathcal F_\tau) \f$ with
///   \f$ W_\beta(F)=\sum_{m\in\mathbb Z}e^{-m^2/(2\beta)}F^m \f$ and
///   \f$ \beta_V=\beta/\langle m^2\rangle_\beta \f$ (see `VillainCharacter`).
///   It is the holonomy term of the whitepaper's action and the default.
/// * `Wilson` — the Wilson plaquette form
///   \f$ S_{\rm hol}=\beta\sum_\tau\bigl(1-\tfrac12(\mathcal F_\tau
///   +\mathcal F_\tau^{-1})\bigr) \f$, the whitepaper's stand-in for the
///   holonomy term. The two forms have the same second variation
///   \f$ \beta L_1^{\rm up} \f$ at trivial holonomy and differ away from it:
///   at \f$ \mathcal F=e^{\pm i\pi/2} \f$ the Wilson curvature vanishes and
///   the Villain curvature does not.
enum class HolonomyForm { Villain, Wilson };

/// # VillainSeries
///
/// The truncated Laurent series of the Villain character sum at one face
/// holonomy \f$ F \f$, and the certified bounds on what the truncation
/// leaves out.
///
/// With \f$ q=e^{-1/(2\beta)} \f$ and \f$ D=F\,d/dF \f$ the Maurer-Cartan
/// derivative, the three sums are
/// \f$ W=\sum_{|m|\le M}q^{m^2}F^m \f$,
/// \f$ DW=\sum_{|m|\le M}m\,q^{m^2}F^m \f$ and
/// \f$ D^2W=\sum_{|m|\le M}m^2q^{m^2}F^m \f$.
///
/// Each tail bound is an upper bound on the modulus of the omitted part of the
/// corresponding infinite series, \f$ \sum_{|m|>M} \f$. With
/// \f$ r=\max(|F|,|F|^{-1}) \f$ every omitted term obeys
/// \f$ |m^k q^{m^2}(F^m+F^{-m})|\le 2t_m \f$ with
/// \f$ t_m=m^k q^{m^2}r^m \f$, and for \f$ m\ge M+1 \f$ the ratio
/// \f$ t_{m+1}/t_m=((m+1)/m)^k q^{2m+1}r \f$ decreases in \f$ m \f$, so it is
/// at most \f$ \rho_k=((M+2)/(M+1))^k q^{2M+3}r \f$ and the tail is at most
/// \f$ 2t_{M+1}/(1-\rho_k) \f$ whenever \f$ \rho_k<1 \f$.
struct VillainSeries {
  /// \f$ W(F) \f$.
  std::complex<double> value{1.0, 0.0};
  /// \f$ F\,W'(F) \f$.
  std::complex<double> first{0.0, 0.0};
  /// \f$ (F\,d/dF)^2W(F) \f$.
  std::complex<double> second{0.0, 0.0};
  /// \f$ M \f$, the largest \f$ |m| \f$ kept.
  std::size_t termCount = 0;
  /// The bound on the omitted part of \f$ W \f$.
  double valueTail = 0.0;
  /// The bound on the omitted part of \f$ DW \f$.
  double firstTail = 0.0;
  /// The bound on the omitted part of \f$ D^2W \f$.
  double secondTail = 0.0;
  /// \f$ \sum_{|m|\le M}|q^{m^2}F^m| \f$, the scale of the rounding error of
  /// the summed \f$ W \f$: a computed \f$ |W| \f$ within a small multiple of
  /// machine epsilon times this is indistinguishable from zero.
  double magnitude = 1.0;
};

/// # VillainCharacter
///
/// The Villain (heat-kernel) weight of one face in its character form, and the
/// per-face potential the joint action's `HolonomyForm::Villain` term sums.
///
/// Reference: Villain, "Theory of one- and two-dimensional magnets with an
/// easy magnetization plane. II", Journal de Physique 36, 581 (1975).
///
/// ## The weight
///
/// For a coupling \f$ \beta>0 \f$ the character sum
/// \f[
///   W_\beta(F)=\sum_{m\in\mathbb Z}e^{-m^2/(2\beta)}F^m
/// \f]
/// is a Laurent series in the face holonomy \f$ F\in\mathbb C^{*} \f$ that
/// converges on all of \f$ \mathbb C^{*} \f$, so it is holomorphic there. Its
/// coefficients are even in \f$ m \f$, so \f$ W(F^{-1})=W(F) \f$. On the unit
/// circle, \f$ F=e^{i\theta} \f$, it is the heat kernel of the circle,
/// \f$ W=\sqrt{2\pi\beta}\sum_n e^{-\beta(\theta-2\pi n)^2/2} \f$ by
/// Poisson summation, which is real and positive. By Jacobi's triple product
/// \f$ W=\prod_{n\ge1}(1-q^{2n})(1+q^{2n-1}F)(1+q^{2n-1}F^{-1}) \f$ with
/// \f$ q=e^{-1/(2\beta)} \f$, so its zeros are exactly the points
/// \f$ F=-q^{\pm(2n-1)} \f$, \f$ n\ge1 \f$, on the negative real axis
/// outside the annulus \f$ q<|F|<q^{-1} \f$.
///
/// ## The potential and its matching to the Wilson form
///
/// The per-face potential is \f$ \phi(F)=-\beta_V\log W_\beta(F) \f$. In the
/// Maurer-Cartan derivative \f$ D=F\,d/dF \f$, which on the unit circle is
/// \f$ -i\,d/d\theta \f$,
/// \f[
///   D\phi=-\beta_V\,\frac{DW}{W},\qquad
///   D^2\phi=-\beta_V\Bigl(\frac{D^2W}{W}-\Bigl(\frac{DW}{W}\Bigr)^2\Bigr).
/// \f]
/// Writing \f$ \langle m^k\rangle_F=D^kW/W \f$ for the moments of the complex
/// weights \f$ e^{-m^2/(2\beta)}F^m/W(F) \f$, the curvature in the real angle,
/// \f$ \partial_\theta^2\phi=-D^2\phi \f$, is
/// \f$ \kappa(F)=\beta_V(\langle m^2\rangle_F-\langle m\rangle_F^2) \f$.
/// At trivial holonomy \f$ \langle m\rangle_1=0 \f$, so
/// \f$ \kappa(1)=\beta_V\langle m^2\rangle_\beta \f$ with
/// \f[
///   \langle m^2\rangle_\beta=\frac{\sum_m m^2e^{-m^2/(2\beta)}}
///                                  {\sum_m e^{-m^2/(2\beta)}} .
/// \f]
/// The Wilson potential \f$ \beta(1-\cos\theta) \f$ has curvature
/// \f$ \beta \f$ there, and the two quadratic expansions agree exactly when
/// \f[
///   \beta_V=\frac{\beta}{\langle m^2\rangle_\beta},
/// \f]
/// which is the matching this class applies. With it the second variation of
/// \f$ \sum_\tau\phi(\mathcal F_\tau) \f$ in the real link angles at trivial
/// holonomy is \f$ \beta L_1^{\rm up} \f$, as for the Wilson form. By
/// Poisson summation
/// \f$ \langle m^2\rangle_\beta=\beta\bigl(1-4\pi^2\beta
/// \langle n^2\rangle\bigr) \f$ with \f$ n \f$ weighted by
/// \f$ e^{-2\pi^2\beta n^2} \f$, so \f$ \beta_V\to1 \f$ as
/// \f$ \beta\to\infty \f$, and \f$ \beta_V-1 \f$ is about \f$ 2\times10^{-3} \f$
/// at \f$ \beta=\tfrac12 \f$ and \f$ 2\times10^{-7} \f$ at
/// \f$ \beta=1 \f$.
///
/// At a quarter-turn holonomy, \f$ F=\pm i \f$, the Wilson curvature
/// \f$ \beta\cos\theta \f$ vanishes while the Villain curvature is
/// \f[
///   \kappa(\pm i)=\beta_V\bigl(\langle m^2\rangle_i-\langle m\rangle_i^2\bigr),
///   \quad
///   \langle m\rangle_i=\frac{2i\sum_{j\ge0}(-1)^j(2j+1)e^{-(2j+1)^2/(2\beta)}}
///                             {\sum_k(-1)^ke^{-2k^2/\beta}},
///   \quad
///   \langle m^2\rangle_i=\frac{\sum_k(-1)^k4k^2e^{-2k^2/\beta}}
///                               {\sum_k(-1)^ke^{-2k^2/\beta}},
/// \f]
/// which is real and positive: \f$ 0.43091 \f$ at \f$ \beta=\tfrac12 \f$,
/// \f$ 0.99796 \f$ at \f$ \beta=1 \f$, \f$ 1.99999958 \f$ at
/// \f$ \beta=2 \f$.
///
/// ## What is evaluated, and where no branch is chosen
///
/// \f$ D\phi \f$ and \f$ D^2\phi \f$ use the ratios \f$ DW/W \f$ and
/// \f$ D^2W/W \f$ only, so the stationarity equations and every derivative are
/// single-valued meromorphic functions of \f$ F \f$ and no logarithm is taken
/// in them. The value \f$ \phi \f$ needs \f$ \log W \f$. On the domain
/// \f$ \Omega=\mathbb C^{*}\setminus\bigl((-\infty,-q^{-1}]\cup[-q,0)\bigr) \f$,
/// which contains every zero-free ray from the unit circle, \f$ W \f$ has no
/// zero and has winding number zero around the unit circle (it is real and
/// positive there), so \f$ \log W \f$ has exactly one holomorphic branch on
/// \f$ \Omega \f$ that is real on the unit circle. `logarithm` evaluates that
/// branch by continuation along the radial path
/// \f$ F(t)=\hat F\,|F|^t \f$, \f$ t\in[0,1] \f$, from
/// \f$ \hat F=F/|F| \f$, accumulating the principal logarithm of the ratio of
/// \f$ W \f$ at consecutive nodes of an adaptive subdivision in which every
/// such ratio, including the one to each interval's midpoint, lies within
/// \f$ \tfrac14 \f$ of one. It throws when the path meets a point at which
/// \f$ |W| \f$ is not certified nonzero, that is, does not exceed a fixed
/// multiple of its tail bound plus its rounding scale, which happens exactly
/// when \f$ F \f$ is at or beyond a zero on the negative real axis.
///
/// ## Truncation
///
/// The series keeps \f$ |m|\le M \f$, where \f$ M_0 \f$ is the least
/// \f$ m\ge1 \f$ with \f$ e^{-m^2/(2\beta)} \f$ below the declared relative
/// tolerance (relative to the \f$ m=0 \f$ coefficient, which is one), and
/// \f$ M\ge M_0 \f$ is raised further, for the holonomy at hand, until the
/// geometric tail ratios \f$ \rho_k \f$ of `VillainSeries` are at most
/// \f$ \tfrac12 \f$ and each of the three tail bounds is below the same
/// tolerance relative to the sum of the moduli of the kept terms of its own
/// series. On the unit circle this is \f$ M_0 \f$; away from it the terms
/// \f$ e^{-m^2/(2\beta)}r^m \f$, \f$ r=\max(|F|,|F|^{-1}) \f$, peak near
/// \f$ m=\beta\log r \f$ and the count grows with them. Every evaluation
/// returns its bounds; nothing is truncated without them.
class VillainCharacter {
 public:
  /// @param beta The coupling \f$ \beta>0 \f$ of the heat kernel, the same
  ///   \f$ \beta \f$ the Wilson form multiplies.
  /// @param tolerance The declared relative tolerance in \f$ (0,1) \f$ below
  ///   which a coefficient \f$ e^{-m^2/(2\beta)} \f$ is left out.
  /// @throws std::invalid_argument when \p beta is not positive or
  ///   \p tolerance is not in \f$ (0,1) \f$.
  VillainCharacter(double beta, double tolerance);

  /// \f$ \beta \f$.
  [[nodiscard]] double beta() const noexcept { return beta_; }
  /// The declared relative tolerance.
  [[nodiscard]] double tolerance() const noexcept { return tolerance_; }
  /// \f$ M_0 \f$, the term count the tolerance alone fixes.
  [[nodiscard]] std::size_t declaredTermCount() const noexcept {
    return declaredTerms_;
  }
  /// \f$ \langle m^2\rangle_\beta \f$, the second moment at trivial holonomy.
  [[nodiscard]] double secondMoment() const noexcept { return secondMoment_; }
  /// \f$ \beta_V=\beta/\langle m^2\rangle_\beta \f$.
  [[nodiscard]] double matchedWeight() const noexcept {
    return beta_ / secondMoment_;
  }

  /// The truncated series \f$ W \f$, \f$ DW \f$, \f$ D^2W \f$ at \p holonomy
  /// with their tail bounds.
  /// @throws std::invalid_argument when \p holonomy is zero or not finite, or
  ///   when no term count up to \f$ 10^5 \f$ meets the truncation rule.
  [[nodiscard]] VillainSeries series(std::complex<double> holonomy) const;

  /// \f$ \log W(F) \f$ on the branch real on the unit circle, by the radial
  /// continuation described above.
  /// @throws std::domain_error when the path meets a point at which \f$ W \f$
  ///   is not certified nonzero.
  [[nodiscard]] std::complex<double> logarithm(
      std::complex<double> holonomy) const;

  /// \f$ \phi(F)=-\beta_V\log W(F) \f$.
  [[nodiscard]] std::complex<double> potential(
      std::complex<double> holonomy) const;
  /// \f$ D\phi=-\beta_V\,DW/W \f$.
  /// @throws std::domain_error when \f$ W(F) \f$ is not certified nonzero.
  [[nodiscard]] std::complex<double> firstDerivative(
      std::complex<double> holonomy) const;
  /// \f$ D^2\phi=-\beta_V(D^2W/W-(DW/W)^2) \f$.
  /// @throws std::domain_error when \f$ W(F) \f$ is not certified nonzero.
  [[nodiscard]] std::complex<double> secondDerivative(
      std::complex<double> holonomy) const;

  /// The relative distance from \p holonomy to the nearest zero of \f$ W \f$,
  /// \f$ \min_{n\ge1,\pm}|F-F_{n,\pm}|/\min(|F|,|F_{n,\pm}|) \f$ over the
  /// zeros \f$ F_{n,\pm}=-e^{\pm(2n-1)/(2\beta)} \f$. It is scale free, as
  /// the zero set is: the zeros accumulate at \f$ 0 \f$ and \f$ \infty \f$,
  /// so an absolute distance would say nothing about the ones near either
  /// end, and dividing by the smaller modulus makes a zero far from \f$ F \f$
  /// in modulus far in this distance too.
  /// @throws std::invalid_argument when \p holonomy is zero or not finite.
  [[nodiscard]] double zeroDistance(std::complex<double> holonomy) const;

 private:
  double beta_;
  double tolerance_;
  double q_;
  std::size_t declaredTerms_ = 1;
  double secondMoment_ = 1.0;
};

/// # HolonomyTruncation
///
/// The truncation of the Villain series over every face of the complex at the
/// current connection, as `JointAction::holonomyTruncation` reports it. For the
/// Wilson form, which is a finite Laurent polynomial, every count and bound is
/// zero.
struct HolonomyTruncation {
  /// The declared form.
  HolonomyForm form = HolonomyForm::Villain;
  /// The declared relative coefficient tolerance.
  double tolerance = 0.0;
  /// \f$ M_0 \f$, the term count the tolerance fixes.
  std::size_t declaredTermCount = 0;
  /// The largest \f$ M \f$ any face needed.
  std::size_t maximumTermCount = 0;
  /// \f$ \max_\tau \text{(tail of } W)/|W(\mathcal F_\tau)| \f$.
  double relativeValueTail = 0.0;
  /// \f$ \max_\tau \text{(tail of } DW)/|W(\mathcal F_\tau)| \f$.
  double relativeFirstTail = 0.0;
  /// \f$ \max_\tau \text{(tail of } D^2W)/|W(\mathcal F_\tau)| \f$.
  double relativeSecondTail = 0.0;
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

  /// \f$ w_R \f$, the coefficient multiplying the Regge action
  /// \f$ S_{\rm Regge}(z) \f$ in the declared `reggeForm`: the primal
  /// \f$ \sum_h|h|\,\varepsilon_h \f$ by default, or the dual
  /// \f$ \sum_h|\!\star\!h|\,\varepsilon_h \f$.
  ///
  /// The whitepaper's Section 7 identification is \f$ w_R = 1/(8\pi G) \f$ in
  /// lattice units, so a caller working in terms of the backreaction coupling
  /// \f$ \kappa = 8\pi G \f$ sets this to \f$ 1/\kappa \f$. The coefficient is
  /// declared here rather than derived, because the units in which the carried
  /// state's bilinear density is measured are the caller's to fix.
  ///
  /// A nonzero weight carries one constraint into any solve of the stationarity
  /// equations. The dual Regge action's exact gradient is analytic on each side
  /// of the real axis in the squared lengths and not across it: the deficit
  /// angle is taken on the principal branch and carries no Riemann-sheet label,
  /// so an arbitrarily small positive imaginary part in \f$ z_e \f$ shifts a
  /// hinge's deficit by \f$ 2\pi \f$ and its contribution to the gradient by
  /// \f$ 2\pi \f$ times the hinge's dual volume. A Jacobian taken on a contour
  /// around a real configuration therefore reads two sheets, and
  /// `HolomorphicJacobianMode::RealAxisDifference` is the rule to declare with
  /// this term.
  double gravitationalWeight = 1.0;

  /// Which Regge discretization \f$ S_{\rm Regge} \f$ is. Primal by default.
  ReggeForm reggeForm = ReggeForm::Primal;

  /// Which hinges the primal sum runs over. Interior by default.
  ReggeHinges reggeHinges = ReggeHinges::Interior;

  /// Which sheet the primal Regge term is read on. Continued by default.
  ReggeBranch reggeBranch = ReggeBranch::Continued;

  /// \f$ z^{(0)}_e \f$, the starting geometry the `Continued` sheets are
  /// continued from, one squared length per edge in `Spacetime::getEdgeList()`
  /// order. Empty means the squared lengths the mesh holds when the
  /// `JointAction` is constructed, which for a relaxation is the geometry it
  /// starts from. Ignored under `ReggeBranch::Principal`.
  std::vector<std::complex<double>> reggeStartSquaredLengths;

  /// \f$ w_S \f$, the coefficient of the linear length stiffness
  /// \f$ S_{\rm stiff}(z)=\tfrac12\sum_e(l_e-l_{0,e})^2 \f$, with
  /// \f$ l_e \f$ the edge length the mesh stores (the root of
  /// \f$ z_e \f$ on the branch the relaxation continues along) and
  /// \f$ l_{0,e} \f$ the declared `referenceLengths`.
  ///
  /// This is the term the whitepaper's own Section 7 computations use in place
  /// of the spectral-moment part of \f$ \mathcal S_0 \f$, which the paper
  /// names but does not write: the energy
  /// \f$ \lambda_0(l)+\tfrac{1}{2\varkappa}\lVert l-l_0\rVert^2 \f$ with the
  /// backreaction coupling \f$ \varkappa \f$ identified as \f$ 8\pi G \f$ in
  /// lattice units. A caller working in terms of \f$ \varkappa \f$ sets this
  /// to \f$ 1/\varkappa \f$. It is a stand-in, labelled as the paper labels
  /// it, and zero leaves it out.
  double stiffnessWeight = 0.0;

  /// \f$ l_{0,e} \f$, one reference length per edge in
  /// `Spacetime::getEdgeList()` order. Required when `stiffnessWeight` is
  /// nonzero; ignored otherwise.
  std::vector<std::complex<double>> referenceLengths;

  /// \f$ \beta \f$, the coupling of the face-holonomy term
  /// \f$ S_{\rm hol}(U) \f$ in the declared `holonomyForm`: the coefficient of
  /// the Wilson form, and the heat-kernel coupling of the Villain form, whose
  /// coefficient is then \f$ \beta_V=\beta/\langle m^2\rangle_\beta \f$. Both
  /// forms have the bare connection stiffness \f$ \beta L_1^{\rm up} \f$ at
  /// trivial holonomy. Zero leaves the term out; the Villain form requires it
  /// to be non-negative.
  double holonomyWeight = 0.0;

  /// Which function of the face holonomies the holonomy term is. Villain by
  /// default.
  HolonomyForm holonomyForm = HolonomyForm::Villain;

  /// The relative tolerance below which a Villain coefficient
  /// \f$ e^{-m^2/(2\beta)} \f$ is left out of the series (`VillainCharacter`).
  /// Ignored by the Wilson form.
  double villainTolerance = 1e-18;

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
/// Reference: Villain, "Theory of one- and two-dimensional magnets with an
/// easy magnetization plane. II", Journal de Physique 36, 581 (1975), for the
/// heat-kernel form.
///
/// The action is the sum of five terms,
/// \f[
///   S(z,U,\Gamma) = w_R\,S_{\rm Regge}(z) + w_S\,S_{\rm stiff}(z)
///                 + w_H\,S_{\rm hol}(U)
///                 + w_M\,S_{\rm matter}(z,U,\Gamma) + S_{\rm spec}(z,U;\xi),
/// \f]
/// in the two edge fields of the microscopic state — the complex squared length
/// \f$ z_e=\ell_e^2 \f$ and the multiplicative connection
/// \f$ U_e \in \mathbb{C}^{*} \f$ — and the carried covariance \f$ \Gamma \f$:
///
/// * \f$ S_{\rm Regge}(z) \f$ is the Regge action in the declared
///   `ReggeForm`: the primal \f$ \sum_h |h|\,\varepsilon_h \f$ over the
///   declared `ReggeHinges`, or the dual Lorentzian
///   \f$ \sum_h |\!\star\! h|\,\varepsilon_h \f$
///   (`simulations::ReggeSolver::dualReggeAction`). Either is a function of
///   the squared lengths alone and independent of \f$ U \f$.
/// * \f$ S_{\rm stiff}(z)=\tfrac12\sum_e(l_e-l_{0,e})^2 \f$ is the linear
///   length stiffness the whitepaper's Section 7 stands in for the
///   spectral-moment part of \f$ \mathcal S_0 \f$.
/// * \f$ S_{\rm hol}(U)=\sum_\tau\phi(\mathcal F_\tau) \f$ runs over the
///   triangles \f$ \tau \f$ of the complex, with the branch-free face holonomy
///   \f$ \mathcal F_\tau=\prod_{e\subset\partial\tau}U_e^{\epsilon_{\tau e}} \f$
///   and \f$ \epsilon_{\tau e} \f$ the integer incidence of \f$ \partial_2 \f$.
///   Under the default `HolonomyForm::Villain` the per-face potential is
///   \f$ \phi(F)=-\beta_V\log W_\beta(F) \f$ with the character sum
///   \f$ W_\beta(F)=\sum_{m\in\mathbb Z}e^{-m^2/(2\beta)}F^m \f$ and
///   \f$ \beta_V=\beta/\langle m^2\rangle_\beta \f$ (`VillainCharacter`);
///   under `HolonomyForm::Wilson`, the whitepaper's stand-in, it is
///   \f$ \phi(F)=\beta\bigl(1-\tfrac12(F+F^{-1})\bigr) \f$, with
///   \f$ \beta \f$ the declared `holonomyWeight` in both. Each is a
///   holomorphic function of \f$ F\in\mathbb C^{*} \f$ (a Laurent series, or
///   a Laurent polynomial) that is even under \f$ F\leftrightarrow F^{-1} \f$,
///   and \f$ \mathcal F_\tau \f$ is a Laurent monomial in the links, so the
///   term is holomorphic on \f$ (\mathbb{C}^{*})^{|E|} \f$ and no argument,
///   logarithm or modulus of a link is ever taken. Evenness makes trivial
///   holonomy stationary, and both forms have the same second variation there:
///   in the real link angles it is \f$ \beta L_1^{\rm up} \f$ with the
///   up-Laplacian \f$ L_1^{\rm up}=\partial_2\partial_2^{\mathsf T} \f$, which
///   vanishes on pure-gauge directions and is \f$ 4\beta \f$ on the coexact
///   block of the regular tetrahedron. They differ away from it: the Wilson
///   curvature \f$ \beta\cos\Theta \f$ vanishes at a quarter-turn holonomy
///   \f$ \mathcal F=e^{\pm i\pi/2} \f$ and the Villain curvature does not.
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
  ///   degree is negative, when a declared moment order is below one, when
  ///   the covariance is present and is not a square matrix over the
  ///   \f$ k \f$-cells of the complex, when the stiffness weight is nonzero
  ///   and the reference lengths do not give one length per edge, or when the
  ///   Villain form is declared with a negative weight or a tolerance outside
  ///   \f$ (0,1) \f$.
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

  /// \f$ w_R\,S_{\rm Regge}(z) \f$ in the declared form and hinge set.
  [[nodiscard]] std::complex<double> reggeTerm() const;
  /// \f$ w_S\,S_{\rm stiff}(z) \f$.
  [[nodiscard]] std::complex<double> stiffnessTerm() const;
  /// \f$ S_{\rm hol}(U) \f$ in the declared form, the weight included. For
  /// the Villain form this is the one quantity that needs \f$ \log W \f$, and
  /// it is evaluated on the branch real on the unit circle
  /// (`VillainCharacter::logarithm`); it throws `std::domain_error` for a face
  /// holonomy at or beyond a zero of \f$ W \f$ on the negative real axis.
  /// `HolomorphicRelaxation` reads the value only to report it: its steps and
  /// their acceptance use the stationarity residual alone.
  [[nodiscard]] std::complex<double> holonomyTerm() const;
  /// \f$ w_M\operatorname{tr}(\Gamma h(z,U)) \f$.
  [[nodiscard]] std::complex<double> matterTerm() const;
  /// \f$ \sum_j \xi_j\,(p_j(h)-p_j^{\star}) \f$.
  [[nodiscard]] std::complex<double> spectralTerm() const;
  /// The whole action \f$ S(z,U,\Gamma) \f$, the sum of the five terms above.
  [[nodiscard]] std::complex<double> value() const;

  /// \f$ \partial S/\partial z_e \f$ for every edge, in `getEdgeList()` order.
  ///
  /// Assembled from the exact analytic gradients the framework supplies: for
  /// the primal Regge form the per-hinge product rule
  /// \f$ \sum_h(\partial|h|\,\varepsilon_h+|h|\,\partial\varepsilon_h) \f$ from
  /// `mesh::Simplex::volumeGradient` and `mesh::Simplex::deficitAngleGradient`,
  /// for the dual form `simulations::ReggeSolver::actionGradientExact`, the
  /// stiffness gradient \f$ w_S(l_e-l_{0,e})/(2l_e) \f$ in closed form, and the
  /// operator gradient `HodgeLaplacian::laplacianGradient`. No finite difference
  /// enters it, and no imaginary part is discarded: the returned number is the
  /// full complex derivative of a holomorphic function.
  [[nodiscard]] std::vector<std::complex<double>> lengthStationarity() const;

  /// \f$ U_e\,\partial S/\partial U_e \f$ for every edge, in `getEdgeList()`
  /// order and on the edge's stored orientation.
  ///
  /// The face-holonomy part is analytic in closed form,
  /// \f$ \sum_\tau \epsilon_{\tau e}\,D\phi(\mathcal F_\tau) \f$ with
  /// \f$ D=F\,d/dF \f$: \f$ -\beta_V\,DW/W \f$ per face for the Villain form
  /// and \f$ -\tfrac12\beta(\mathcal F_\tau-\mathcal F_\tau^{-1}) \f$ for the
  /// Wilson form. The operator part uses the
  /// identity \f$ U_e\,\partial/\partial U_e = -i\,\partial/\partial\varphi_e \f$
  /// for \f$ U_e=e^{i\varphi_e} \f$ together with the exact analytic
  /// `HodgeLaplacian::laplacianPhaseGradient`; that identity is a relation
  /// between derivatives and selects no branch of the logarithm.
  [[nodiscard]] std::vector<std::complex<double>> linkStationarity() const;

  /// The Maurer-Cartan Hessian of the face-holonomy term,
  /// \f$ U_e\partial_{U_e}\bigl(U_{e'}\partial_{U_{e'}}w_HS_{\rm hol}\bigr)
  ///   =\sum_\tau\epsilon_{\tau e}\epsilon_{\tau e'}\,D^2\phi(\mathcal F_\tau) \f$,
  /// flat row-major \f$ |E|\times|E| \f$ in `getEdgeList()` order on the
  /// stored orientations.
  ///
  /// This is the holonomy term's exact contribution to the link block of the
  /// Jacobian of `linkStationarity` in the multiplicative coordinate
  /// \f$ U\mapsto Ue^{\delta} \f$ that `HolomorphicRelaxation` steps in. In
  /// the real angles, \f$ \delta=i\theta \f$, the Hessian is its negative. The
  /// per-face second derivative is \f$ -\beta_V(D^2W/W-(DW/W)^2) \f$ for the
  /// Villain form and \f$ -\tfrac12\beta(\mathcal F+\mathcal F^{-1}) \f$ for
  /// the Wilson form; no logarithm enters either.
  [[nodiscard]] std::vector<std::complex<double>> holonomyHessian() const;

  /// The truncation of the Villain series over the current face holonomies,
  /// with its certified relative tail bounds. All zero for the Wilson form.
  [[nodiscard]] HolonomyTruncation holonomyTruncation() const;

  /// The smallest relative distance of any face holonomy
  /// \f$ \mathcal F_\tau \f$ to a zero of \f$ W \f$
  /// (`VillainCharacter::zeroDistance`). Positive infinity when the declared
  /// term has no zero: the Wilson form, or a zero weight.
  [[nodiscard]] double holonomyZeroDistance() const;

  /// The smallest relative distance to a zero of \f$ W \f$ that any face
  /// holonomy comes to along the multiplicative path
  /// \f$ U_e\mapsto U_e\,e^{t\delta_e} \f$, \f$ t\in[0,1] \f$, from the
  /// current connection.
  ///
  /// Along the path each face moves as
  /// \f$ \mathcal F_\tau(t)=\mathcal F_\tau\,e^{t\Delta_\tau} \f$ with
  /// \f$ \Delta_\tau=\sum_e\epsilon_{\tau e}\delta_e \f$, formed from the
  /// increments without any logarithm. The distance is sampled at nodes spaced
  /// by at most \p spacing in \f$ |t\Delta_\tau| \f$, the Maurer-Cartan
  /// length of the path, both end points included, so a path that passes a
  /// zero between nodes still reads a distance of order \p spacing there.
  ///
  /// @param linkIncrements \f$ \delta_e \f$ per edge, in `getEdgeList()` order
  ///   and on the stored orientations, as `HolomorphicRelaxation` steps them.
  /// @param spacing The largest node spacing, positive.
  /// @return Positive infinity when the declared term has no zero.
  /// @throws std::invalid_argument when the increment count is not the edge
  ///   count or the spacing is not positive.
  [[nodiscard]] double holonomyZeroClearance(
      const std::vector<std::complex<double>> &linkIncrements,
      double spacing) const;

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

  /// The Ward current on the canonical degree-one cells, in the canonical
  /// `ChainComplex::kSimplexVertices(1)` order and on each cell's canonical
  /// ascending-vertex orientation.
  ///
  /// `wardCurrent` reports the current on the mesh's stored edge orientation
  /// and in `Spacetime::getEdgeList()` order, which is the order the rest of
  /// the engine's per-edge quantities use. Every chain-level consumer — the
  /// boundary map, a cut's coorientation, a restriction to a set of cells —
  /// indexes cells canonically instead, and the current is odd under reversing
  /// an edge, so the two differ by a reordering and a per-edge sign. This
  /// method applies both. An edge of the mesh that the chain complex does not
  /// carry contributes nothing, and a canonical cell no mesh edge matches
  /// stays zero.
  [[nodiscard]] std::vector<std::complex<double>> canonicalWardCurrent() const;

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

  /// The number of hinges the primal Regge sum runs over under the declared
  /// hinge rule, reported so a caller can see when the primal term is empty
  /// (a complex with no interior hinge under `ReggeHinges::Interior`).
  [[nodiscard]] std::size_t reggeHingeCount() const;

  /// True when the Regge term is declared (nonzero weight, primal form) and
  /// the complex has no hinge under the declared hinge rule, so that the term
  /// and its gradient are identically zero for every geometry. This is a
  /// property of the complex, not of its lengths: under `ReggeHinges::Interior`
  /// it holds on a complex none of whose hinges has a closed link, such as a
  /// flag complex in which every triangle lies in more than two tetrahedra.
  [[nodiscard]] bool reggeStructurallyZero() const;

  /// Under `ReggeBranch::Continued`, the number of dihedral angles of the
  /// primal sum whose continued sheet differs from the principal one at the
  /// current geometry: a nonzero count is the number of angles the principal
  /// evaluation would have put on another sheet. Zero under
  /// `ReggeBranch::Principal`.
  [[nodiscard]] std::size_t reggeOffPrincipalAngles() const;

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

  /// The names of the five declared terms, in the order `value` sums them:
  /// `"regge"`, `"stiffness"`, `"holonomy"`, `"matter"`, `"spectral"`.
  [[nodiscard]] static std::vector<std::string> termNames();

 private:
  /// The declared sheets of the primal Regge sum at the current geometry: per
  /// hinge, the sheets of its dihedral angles and the sign of its content root
  /// relative to the principal one.
  struct ReggeSheets;
  [[nodiscard]] ReggeSheets reggeSheets() const;

  std::shared_ptr<Spacetime> spacetime_;
  JointActionDeclaration declaration_;
  /// \f$ z^{(0)} \f$ keyed by the edge's sorted vertex ids.
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
      reggeStart_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_JOINTACTION_H
