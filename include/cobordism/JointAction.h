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

/// # SpectralConstraintForm
///
/// What one `SpectralMomentConstraint` pins.
enum class SpectralConstraintForm {
  /// The power sum \f$ p_j(h)=\operatorname{tr}(h^j) \f$ of the carrier, or
  /// of its compression to the declared fiber.
  PowerSum,
  /// The mean eigenvalue of one band,
  /// \f$ \lambda_b=\operatorname{tr}(P_bh)/r_b \f$, with \f$ P_b \f$ the
  /// band's Riesz projector (`JointActionDeclaration::momentBandProjectors`)
  /// and \f$ r_b=\operatorname{tr}P_b \f$ its rank: the band's eigenvalue
  /// itself when the band is degenerate, as every band of a sheeted host is.
  /// Its derivative at fixed \f$ P_b \f$ is the Hellmann-Feynman form
  /// \f$ d\lambda_b=\operatorname{tr}(P_b\,dh)/r_b \f$, the whole derivative
  /// when \f$ P_b \f$ is a spectral projector of \f$ h \f$: the projector's
  /// own variation is off-diagonal between its range and its kernel and so
  /// contributes no trace against \f$ h \f$.
  BandMean
};

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
  /// What the constraint pins: a power sum (the default) or a band mean.
  SpectralConstraintForm form = SpectralConstraintForm::PowerSum;
  /// The moment index \f$ j \ge 1 \f$ of a power sum. It is the power the
  /// operator is raised to in \f$ p_j(h)=\operatorname{tr}(h^j) \f$ and is
  /// not a simplicial degree. Unused by a band mean.
  int order = 1;
  /// For a band mean, the index of the band's projector in
  /// `JointActionDeclaration::momentBandProjectors`. Unused by a power sum.
  std::size_t band = 0;
  /// The target: \f$ p_j^{\star}=\sum_a (\lambda_a^{\star})^j \f$, the same
  /// power sum of the prescribed eigenvalue multiset, or the band mean's
  /// \f$ \lambda_b^{\star} \f$.
  std::complex<double> target{0.0, 0.0};
  /// The complex Lagrange multiplier \f$ \xi_j \f$ at the current point of a
  /// solve.
  std::complex<double> multiplier{0.0, 0.0};
};

/// # ActionTermGradient
///
/// One term of the joint action at the action's point: its value and its
/// contribution to the stationarity equations,
/// \f$ \partial S_{\rm term}/\partial z_e \f$ on the lengths and
/// \f$ U_e\,\partial S_{\rm term}/\partial U_e \f$ on the links, one entry
/// per edge of the complex each. Summed over the terms these are
/// `JointAction::lengthStationarity` and `JointAction::linkStationarity`.
struct ActionTermGradient {
  /// `"regge"`, `"holonomy"`, `"matter"`, or
  /// `"constraint j"` with \f$ j \f$ counted from one in declaration order.
  std::string name;
  /// The term as it stands in the action: `"(1/kappa) S_Regge"`,
  /// `"beta S_hol"`, `"w_m tr(Gamma h_1)"`, or
  /// `"xi_j (c_j - c_j*)"` followed by what \f$ c_j \f$ is, the band
  /// eigenvalue \f$ \lambda_b/s \f$ or the power sum \f$ p_j(h_C/s) \f$.
  std::string label;
  /// The coefficient in front of the term: \f$ w_R=1/\kappa \f$,
  /// \f$ w_S \f$, \f$ \beta \f$, \f$ w_m \f$, or the multiplier
  /// \f$ \xi_j \f$.
  std::complex<double> weight{0.0, 0.0};
  /// What the weight multiplies: \f$ S_{\rm Regge} \f$,
  /// \f$ \operatorname{tr}(\Gamma h) \f$, or the constraint's residual
  /// \f$ c_j-c_j^{\star} \f$ in the declared unit, so that
  /// `value == weight * bare` when `factored`. For the holonomy term, whose
  /// weight enters its form (the Villain weight is \f$ \beta \f$ over the
  /// mean of \f$ m^2 \f$), `bare` is the value itself and `factored` is
  /// false. Zero when the weight is zero and the term was not formed.
  std::complex<double> bare{0.0, 0.0};
  bool factored = true;
  /// The term's value: \f$ w_R S_{\rm Regge} \f$,
  /// \f$ S_{\rm hol} \f$ (quiet NaN where it refuses to evaluate),
  /// \f$ w_m\operatorname{tr}(\Gamma h) \f$, or
  /// \f$ \xi_j\,(c_j-c_j^{\star}) \f$.
  std::complex<double> value{0.0, 0.0};
  std::vector<std::complex<double>> lengthStationarity;
  std::vector<std::complex<double>> linkStationarity;
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

/// # VillainSeries
///
/// The order-\f$ M \f$ Villain sums at one face holonomy \f$ F \f$, the bound
/// on the rounding of the first of them, and the reported distance of each
/// from its infinite series.
///
/// With \f$ c_m \f$ the coefficients of `VillainCharacter`, the
/// double-precision values of \f$ q^{m^2} \f$ with \f$ q=e^{-1/(2\beta)} \f$,
/// and \f$ D=F\,d/dF \f$ the Maurer-Cartan derivative, the three sums are
/// \f$ W_M=\sum_{|m|\le M}c_mF^m \f$,
/// \f$ DW_M=\sum_{|m|\le M}m\,c_mF^m \f$ and
/// \f$ D^2W_M=\sum_{|m|\le M}m^2c_mF^m \f$: a Laurent polynomial and its two
/// derivatives, each summed in full and exact up to rounding.
///
/// Each tail is an upper bound on the modulus of the part of the
/// corresponding infinite series beyond the order, \f$ \sum_{|m|>M} \f$: the
/// distance between the order-\f$ M \f$ sum and the infinite series. With
/// \f$ r=\max(|F|,|F|^{-1}) \f$ every pair beyond the order obeys
/// \f$ |m^k q^{m^2}(F^m+F^{-m})|\le 2t_m \f$ with
/// \f$ t_m=m^k q^{m^2}r^m \f$, and the ratio
/// \f$ t_{m+1}/t_m=((m+1)/m)^k q^{2m+1}r \f$ decreases in \f$ m \f$, so past
/// an index \f$ N \f$ it is at most
/// \f$ \rho_k(N)=((N+2)/(N+1))^k q^{2N+3}r \f$. With \f$ N\ge M \f$ the least
/// index at which \f$ \rho_k(N)<1 \f$, the tail is at most
/// \f[
///   \sum_{m=M+1}^{N}2t_m+\frac{2t_{N+1}}{1-\rho_k(N)} .
/// \f]
/// Where \f$ \rho_k(M)<1 \f$, \f$ N=M \f$ and the bound is the geometric one
/// alone. The tails are a report: no sum, no certificate and no step of a
/// solve reads them.
struct VillainSeries {
  /// \f$ W_M(F) \f$.
  std::complex<double> value{1.0, 0.0};
  /// \f$ F\,W_M'(F) \f$.
  std::complex<double> first{0.0, 0.0};
  /// \f$ (F\,d/dF)^2W_M(F) \f$.
  std::complex<double> second{0.0, 0.0};
  /// \f$ M \f$, the declared order: the largest \f$ |m| \f$ in the sums.
  int order = 0;
  /// The bound on \f$ |W-W_M| \f$, \f$ W \f$ the infinite series.
  double valueTail = 0.0;
  /// The bound on \f$ |DW-DW_M| \f$.
  double firstTail = 0.0;
  /// The bound on \f$ |D^2W-D^2W_M| \f$.
  double secondTail = 0.0;
  /// \f$ A(F)=\sum_{|m|\le M}c_m|F|^m \f$, the sum of the moduli of the terms
  /// of \f$ W_M \f$, evaluated with every operation rounded upward, so that
  /// it is not below the exact sum.
  double magnitude = 1.0;
  /// \f$ \sum_{|m|\le M}|m|\,c_m|F|^m \f$, the same for \f$ DW_M \f$: a bound
  /// on \f$ |DW_M(F)| \f$.
  double firstMagnitude = 0.0;
  /// \f$ \sum_{|m|\le M}m^2c_m|F|^m \f$, the same for \f$ D^2W_M \f$.
  double secondMagnitude = 0.0;
  /// The bound on the rounding of `value`,
  /// \f$ |{\rm fl}(W_M)-W_M|\le\gamma_{6M+1}A(F) \f$, derived under
  /// `VillainCharacter`; infinite when the computed sum is not finite.
  double roundingBound = 0.0;
};

/// # VillainCharacter
///
/// The Villain (heat-kernel) weight of one face in its character form, summed
/// to a declared order, and the per-face potential the joint action's holonomy
/// term sums.
///
/// Reference: Villain, "Theory of one- and two-dimensional magnets with an
/// easy magnetization plane. II", Journal de Physique 36, 581 (1975).
/// Reference: Higham, "Accuracy and Stability of Numerical Algorithms", second
/// edition, SIAM (2002), Lemmas 3.1, 3.3 and 3.5, for the rounding bounds.
///
/// ## The weight and its order
///
/// For a coupling \f$ \beta>0 \f$ the character sum
/// \f[
///   W_\beta(F)=\sum_{m\in\mathbb Z}e^{-m^2/(2\beta)}F^m
/// \f]
/// is a Laurent series in the face holonomy \f$ F\in\mathbb C^{*} \f$ that
/// converges on all of \f$ \mathbb C^{*} \f$. It is a Jacobi theta function
/// and has no closed elementary form, so it is summed to a declared order
/// \f$ M \f$, an integer from one to ten (`maximumOrder`):
/// \f[
///   W_M(F)=\sum_{|m|\le M}c_mF^m ,
/// \f]
/// with \f$ c_m \f$ the double-precision value of \f$ e^{-m^2/(2\beta)} \f$
/// (`coefficients`). The holonomy term of the action is defined by this
/// Laurent polynomial: every value, derivative and certificate below is that
/// of \f$ W_M \f$, and the infinite series enters only through the reported
/// tails of `VillainSeries`.
///
/// The coefficients are even in \f$ m \f$ and real, so
/// \f$ W_M(F^{-1})=W_M(F) \f$ and
/// \f$ W_M(\bar F)=\overline{W_M(F)} \f$. On the unit circle,
/// \f$ F=e^{i\theta} \f$, \f$ W_M=1+2\sum_{m=1}^{M}c_m\cos m\theta \f$ is
/// real. \f$ F^MW_M(F) \f$ is a polynomial of degree \f$ 2M \f$, so
/// \f$ W_M \f$ has \f$ 2M \f$ zeros in \f$ \mathbb C^{*} \f$, a set closed
/// under inversion and under conjugation.
///
/// The infinite series is, on the unit circle, the heat kernel of the circle,
/// \f$ W=\sqrt{2\pi\beta}\sum_n e^{-\beta(\theta-2\pi n)^2/2} \f$ by Poisson
/// summation, real and positive, and by Jacobi's triple product
/// \f$ W=\prod_{n\ge1}(1-q^{2n})(1+q^{2n-1}F)(1+q^{2n-1}F^{-1}) \f$ with
/// \f$ q=e^{-1/(2\beta)} \f$ its zeros are exactly the points
/// \f$ F=-q^{\pm(2n-1)} \f$, \f$ n\ge1 \f$. \f$ W_M \f$ has these properties
/// only to within its tail: on the unit circle it differs from the heat
/// kernel by at most the tail bound and the rounding of the coefficients, so
/// it is positive where the heat kernel is larger than that, and where the
/// heat kernel is smaller it need not be.
///
/// ## The potential and its matching to the Wilson form
///
/// The per-face potential is \f$ \phi(F)=-\beta_V\log W_M(F) \f$. In the
/// Maurer-Cartan derivative \f$ D=F\,d/dF \f$, which on the unit circle is
/// \f$ -i\,d/d\theta \f$,
/// \f[
///   D\phi=-\beta_V\,\frac{DW_M}{W_M},\qquad
///   D^2\phi=-\beta_V\Bigl(\frac{D^2W_M}{W_M}
///                        -\Bigl(\frac{DW_M}{W_M}\Bigr)^2\Bigr).
/// \f]
/// Writing \f$ \langle m^k\rangle_F=D^kW_M/W_M \f$ for the moments of the
/// complex weights \f$ c_mF^m/W_M(F) \f$, \f$ |m|\le M \f$, the curvature in
/// the real angle, \f$ \partial_\theta^2\phi=-D^2\phi \f$, is
/// \f$ \kappa(F)=\beta_V(\langle m^2\rangle_F-\langle m\rangle_F^2) \f$.
/// At trivial holonomy \f$ \langle m\rangle_1=0 \f$, so
/// \f$ \kappa(1)=\beta_V\langle m^2\rangle_\beta \f$ with the second moment of
/// the same order-\f$ M \f$ sums,
/// \f[
///   \langle m^2\rangle_\beta=\frac{\sum_{|m|\le M}m^2c_m}
///                                  {\sum_{|m|\le M}c_m} .
/// \f]
/// The Wilson potential \f$ \beta(1-\cos\theta) \f$ has curvature
/// \f$ \beta \f$ there, and the two quadratic expansions agree exactly when
/// \f[
///   \beta_V=\frac{\beta}{\langle m^2\rangle_\beta},
/// \f]
/// which is the matching this class applies, with the order-\f$ M \f$ moment.
/// With it the second variation of \f$ \sum_\tau\phi(\mathcal F_\tau) \f$ in
/// the real link angles at trivial holonomy is \f$ \beta L_1^{\rm up} \f$, as
/// for the Wilson form, at every order. For the infinite series Poisson
/// summation gives
/// \f$ \langle m^2\rangle_\beta=\beta\bigl(1-4\pi^2\beta
/// \langle n^2\rangle\bigr) \f$ with \f$ n \f$ weighted by
/// \f$ e^{-2\pi^2\beta n^2} \f$, so its \f$ \beta_V-1 \f$ is about
/// \f$ 2\times10^{-3} \f$ at \f$ \beta=\tfrac12 \f$ and
/// \f$ 2\times10^{-7} \f$ at \f$ \beta=1 \f$, and the order-ten sums reproduce
/// these to rounding. At a fixed order the coefficients tend to one as
/// \f$ \beta\to\infty \f$, so \f$ \langle m^2\rangle_\beta \f$ tends to
/// \f$ M(M+1)/3 \f$ and \f$ \beta_V \f$ grows as \f$ 3\beta/(M(M+1)) \f$; the
/// reported tail says how far a coupling is from the infinite series.
///
/// At a quarter-turn holonomy, \f$ F=\pm i \f$, the Wilson curvature
/// \f$ \beta\cos\theta \f$ vanishes while the Villain curvature is
/// \f$ \kappa(\pm i)=\beta_V\bigl(\langle m^2\rangle_i
/// -\langle m\rangle_i^2\bigr) \f$, in which only the even \f$ m \f$ enter
/// \f$ W_M(i) \f$ and \f$ \langle m^2\rangle_i \f$, with the sign
/// \f$ (-1)^{m/2} \f$, and only the odd \f$ m \f$ enter
/// \f$ \langle m\rangle_i \f$. It is real and positive: at order ten
/// \f$ 0.43091 \f$ at \f$ \beta=\tfrac12 \f$, \f$ 0.99796 \f$ at
/// \f$ \beta=1 \f$ and \f$ 1.99999958 \f$ at \f$ \beta=2 \f$.
///
/// ## What is evaluated, and where no branch is chosen
///
/// \f$ D\phi \f$ and \f$ D^2\phi \f$ use the ratios \f$ DW_M/W_M \f$ and
/// \f$ D^2W_M/W_M \f$ only, so the stationarity equations and every
/// derivative are single-valued rational functions of \f$ F \f$ and no
/// logarithm is taken in them. They are evaluated wherever \f$ W_M \f$ is
/// certified nonzero. The value \f$ \phi \f$ needs \f$ \log W_M \f$, which
/// `logarithm` evaluates by a certified continuation.
///
/// ## The rounding bound and the certificate that the weight is nonzero
///
/// \f$ W_M \f$ is the Laurent polynomial with exactly the coefficients
/// \f$ c_m \f$, so the coefficients carry no error. `series` evaluates
/// \f$ W_M(F)=1+\sum_{m=1}^{M}c_m(F^m+G^m) \f$, \f$ G=1/F \f$, in double
/// precision in the order written: the \f$ n=2M+1 \f$ monomials are added
/// pair by pair in increasing \f$ m \f$. Let \f$ u=2^{-53} \f$ be the unit
/// roundoff and \f$ \gamma_k=ku/(1-ku) \f$. A product of \f$ k \f$ factors
/// \f$ 1+\delta_i \f$ with \f$ |\delta_i|\le u \f$ is \f$ 1+\theta \f$ with
/// \f$ |\theta|\le\gamma_k \f$, and
/// \f$ (1+\gamma_j)(1+\gamma_k)\le1+\gamma_{j+k} \f$ (Higham, Lemmas 3.1 and
/// 3.3). Each computed monomial is the exact one times such a product, whose
/// factors are counted as follows.
///
/// * The inverse. \f$ F=s\,2^e \f$ is scaled by an exact power of two to
///   \f$ \tfrac12\le\max(|{\rm Re}\,s|,|{\rm Im}\,s|)<1 \f$ and
///   \f$ G=\bar s/|s|^2\cdot2^{-e} \f$: two squares and their sum for
///   \f$ |s|^2 \f$ and one division for each component, three roundings on
///   each component, so the computed inverse is \f$ G(1+\theta) \f$ with
///   \f$ |\theta|\le\gamma_3 \f$.
/// * The powers. A complex product formed as \f$ (ac-bd,\ ad+bc) \f$ has a
///   relative error of at most \f$ \sqrt2\,\gamma_2\le\gamma_3 \f$ (Higham,
///   Lemma 3.5). \f$ F^m \f$ takes \f$ m-1 \f$ products, \f$ 3(m-1) \f$
///   units; \f$ G^m \f$ takes \f$ m-1 \f$ products and carries the computed
///   inverse \f$ m \f$ times, \f$ 6m-3 \f$ units.
/// * The term. The sum \f$ F^m+G^m \f$ and its product with \f$ c_m \f$ are
///   one unit each.
/// * The accumulation. The pair of index \f$ m \f$ passes through
///   \f$ M-m+1 \f$ additions.
///
/// The largest count is that of \f$ c_MG^M \f$, \f$ (6M-3)+2+1=6M \f$, so
/// \f[
///   |{\rm fl}(W_M)-W_M|\le\gamma_{6M}\,A(F),\qquad
///   A(F)=\sum_{|m|\le M}c_m|F|^m .
/// \f]
/// Gradual underflow adds to a real product or quotient an absolute error of
/// at most half the least subnormal number \f$ \eta=2^{-1074} \f$. The
/// evaluation holds \f$ 10M \f$ real products and quotients (eight in the
/// inverse, its four scalings by a power of two included, \f$ 8(M-1) \f$ in
/// the powers and \f$ 2M \f$ in the terms); the
/// error of one reaches at most \f$ M \f$ monomials and adds to each at most
/// \f$ M\,A(F) \f$ times its size, since the later factors of a monomial,
/// times its coefficient, are bounded by \f$ A(F) \f$ and a monomial holds
/// the inverse at most \f$ M \f$ times. Together they add at most
/// \f$ 5M^3\eta\,A(F) \f$, which is below \f$ u\,A(F) \f$ at every order
/// because \f$ \eta/u=2^{-1021} \f$. One more unit covers it, and the
/// rounding bound of `VillainSeries` is
/// \f[
///   \gamma_{6M+1}\,A(F),
/// \f]
/// with \f$ A(F) \f$ and the product evaluated with every operation rounded
/// upward (`std::nextafter`), so that the stored bound is not below the exact
/// one. A sum that overflows has no finite value and its bound is infinite.
///
/// \f$ W_M \f$ is certified nonzero at \f$ F \f$ when the modulus of the
/// computed sum, rounded downward, exceeds this bound: then
/// \f$ W_M(F)\neq0 \f$. No other factor enters the certificate. Where it does
/// not hold the computed sum cannot be told from zero, and \f$ DW_M/W_M \f$,
/// \f$ D^2W_M/W_M \f$ and \f$ \log W_M \f$ have no certified value there
/// (`std::domain_error`).
///
/// ## The logarithm
///
/// `logarithm` evaluates \f$ \log W_M(F) \f$ on the branch that is real at
/// the point \f$ F/|F| \f$ of the unit circle, continued along the radial
/// path from that point to \f$ F \f$. The branch has a value when
/// \f$ W_M \f$ is positive at that point of the circle and the path meets no
/// zero of \f$ W_M \f$.
///
/// The nodes of the path are computed as \f$ P(s)={\rm fl}(F\,s) \f$, the two
/// components each multiplied by a real scale \f$ s \f$ between
/// \f$ s_0={\rm fl}(1/|F|) \f$ and one, with \f$ P(1)=F \f$ itself. The point
/// of the ray a node stands for is \f$ Z(s)=F\,s \f$. Each component of a
/// node is one rounded product, so
/// \f$ |P(s)-Z(s)|\le u\,|Z(s)|+\eta/\sqrt2 \f$ with \f$ \eta \f$ the least
/// subnormal number, and \f$ |Z(s)|\ge\min(|F|,|F|s_0) \f$ on the path, so
/// the relative distance is at most
/// \f$ \omega=u+\eta/\min(|F|,|F|s_0) \f$ and the Maurer-Cartan distance
/// \f$ |\log(P/Z)| \f$ is at most \f$ \varepsilon=\omega/(1-\omega) \f$.
///
/// A step from the node at scale \f$ s_a \f$ to the node at \f$ s_b \f$ is
/// certified by a bound on the change of \f$ W_M \f$. Let \f$ \mathcal N \f$
/// be the set of points within Maurer-Cartan distance \f$ \varepsilon \f$ of
/// the stretch of the ray between \f$ Z(s_a) \f$ and \f$ Z(s_b) \f$; it
/// contains both nodes and is convex in the coordinate \f$ \log F \f$. With
/// \f$ s_- \f$ and \f$ s_+ \f$ the smaller and the larger of the two scales,
/// the moduli in \f$ \mathcal N \f$ lie between
/// \f$ \rho_-=|F|s_-e^{-\varepsilon} \f$ and
/// \f$ \rho_+=|F|s_+e^{\varepsilon} \f$, and the modulus of each monomial is
/// monotone in \f$ |F| \f$, increasing for \f$ m>0 \f$ and decreasing for
/// \f$ m<0 \f$, so on \f$ \mathcal N \f$
/// \f[
///   |D^2W_M|\le B=\sum_{m=1}^{M}m^2c_m\bigl(\rho_+^{\,m}+\rho_-^{-m}\bigr).
/// \f]
/// Every point \f$ G \f$ of \f$ \mathcal N \f$ is joined to \f$ P(s_a) \f$
/// inside \f$ \mathcal N \f$ by the straight segment in the coordinate
/// \f$ \log F \f$, of length at most
/// \f$ L=\log(s_+/s_-)+2\varepsilon\le(s_+-s_-)/s_-+2\varepsilon \f$. Along
/// it \f$ dW_M=DW_M\,d\log F \f$ and \f$ dDW_M=D^2W_M\,d\log F \f$, so by
/// Taylor's formula with its remainder
/// \f[
///   |W_M(G)-W_M(P(s_a))|\le V=|DW_M(P(s_a))|\,L+\tfrac12BL^2 .
/// \f]
/// The first derivative at the node is bounded by the modulus of its computed
/// sum plus that sum's rounding bound, which is derived as that of
/// \f$ W_M \f$ with one more unit for the product \f$ m\,c_m \f$:
/// \f$ \gamma_{6M+2}\sum_{|m|\le M}|m|\,c_m|F|^m \f$. The bound is of second
/// order, with the first-order term read from the computed derivative,
/// because where \f$ W_M \f$ is small beside the terms it is summed from,
/// \f$ DW_M \f$ is small with it while the sum of the moduli of its terms is
/// not: a bound from that sum alone certifies only steps of length
/// \f$ |W_M|/\sum|m|\,c_m|F|^m \f$. With \f$ \hat W_a \f$, \f$ \hat W_b \f$
/// the computed sums at the two nodes and \f$ u_a \f$, \f$ u_b \f$ their
/// rounding bounds, the step is certified when
/// \f[
///   V+u_a+u_b<|\hat W_a| .
/// \f]
/// Then \f$ W_M(\mathcal N) \f$, \f$ \hat W_a \f$ and \f$ \hat W_b \f$ all
/// lie in the open disc of radius \f$ |\hat W_a| \f$ about \f$ \hat W_a \f$.
/// The disc does not contain zero, so \f$ W_M \f$ has no zero on the stretch,
/// and it lies in the half-plane
/// \f$ \{z:\operatorname{Re}(z/\hat W_a)>0\} \f$, on which the principal
/// logarithm of \f$ z/\hat W_a \f$ is a branch of \f$ \log z \f$: the
/// increment of the continued logarithm over the step is the principal
/// logarithm of \f$ \hat W_b/\hat W_a \f$, which cannot wind. The term
/// \f$ u_b \f$ is there because the increment is formed from the computed sum
/// at the far node. Consecutive steps share a node, and the half-planes of
/// both contain the computed sum there and the value of \f$ W_M \f$ at the
/// node's point of the ray, so the increments add up to the logarithm
/// continued along the ray, to within the rounding of the sum at \f$ F \f$.
/// Every quantity of the inequality is evaluated in real arithmetic with each
/// operation rounded to the side that makes the inequality harder to meet
/// (`std::nextafter`), so the computed inequality implies the exact one.
///
/// The first candidate for \f$ s_b \f$ is one, the whole remaining path. A
/// candidate that is not certified is replaced by the geometric mean of
/// \f$ s_a \f$ and \f$ s_b \f$, which halves the Maurer-Cartan length of the
/// segment, until a step is certified; no count bounds the halvings. When the
/// geometric mean is not strictly between \f$ s_a \f$ and \f$ s_b \f$ in
/// double precision the segment cannot be subdivided: the path meets a zero
/// of \f$ W_M \f$ at the resolution of the datatype, the logarithm has no
/// value, and `logarithm` throws `std::domain_error`.
///
/// The start of the path is certified by the same bound. \f$ Z(s_0) \f$ lies
/// within \f$ \lambda=\max(|F|s_0-1,\,1/(|F|s_0)-1) \f$ of the unit circle in
/// the Maurer-Cartan distance, because
/// \f$ |\log x|\le\max(x-1,\,1/x-1) \f$, with \f$ |F| \f$ bounded below and
/// above by its modulus evaluated with every operation rounded downward and
/// upward. The points of that stretch of the ray are within
/// \f$ L_0=\lambda+\varepsilon \f$ of the first node, so with \f$ B_0 \f$
/// the sum above at \f$ \rho_+=\max(1,|F|s_0)\,e^{\varepsilon} \f$ and
/// \f$ \rho_-=\min(1,|F|s_0)\,e^{-\varepsilon} \f$, \f$ W_M \f$ on the
/// stretch, its point on the circle included, differs from the computed sum
/// \f$ \hat W_0 \f$ at the first node by at most
/// \f$ \hat u_0=|DW_M(P(s_0))|\,L_0+\tfrac12B_0L_0^2+u_0 \f$. \f$ W_M \f$
/// is real on the circle, so
/// \f$ |\operatorname{Im}\hat W_0|\le\hat u_0 \f$ holds whenever the
/// evaluation is correct; a larger imaginary part contradicts the reality of
/// the cosine sum and throws `std::logic_error`. When
/// \f$ \operatorname{Re}\hat W_0>\hat u_0 \f$, \f$ W_M \f$ is positive at the
/// point of the circle and the continuation starts from the real logarithm
/// of \f$ \operatorname{Re}\hat W_0 \f$. Otherwise \f$ W_M \f$ is zero,
/// negative or below its bound at that point of the circle, the branch real
/// on the circle has no value on the ray, and `logarithm` throws
/// `std::domain_error`. At the end of the path \f$ W_M \f$ is required to be
/// certified nonzero at \f$ F \f$, and a scale \f$ s_0 \f$ that is not a
/// finite positive double (a holonomy whose modulus or inverse modulus is
/// beyond the range of the datatype) leaves the path without a first node,
/// which is a `std::domain_error` as well.
class VillainCharacter {
 public:
  /// The largest order that can be declared.
  static constexpr int maximumOrder = 10;

  /// @param beta The coupling \f$ \beta>0 \f$ of the heat kernel.
  /// @param order The order \f$ M \f$: the sums keep \f$ |m|\le M \f$.
  /// @throws std::invalid_argument when \p beta is not positive and finite
  ///   or \p order is not an integer from one to `maximumOrder`.
  VillainCharacter(double beta, int order);

  /// \f$ \beta \f$.
  [[nodiscard]] double beta() const noexcept { return beta_; }
  /// \f$ M \f$, the declared order.
  [[nodiscard]] int order() const noexcept { return order_; }
  /// \f$ c_0,\dots,c_M \f$: the double-precision values of
  /// \f$ e^{-m^2/(2\beta)} \f$ that define \f$ W_M \f$.
  [[nodiscard]] const std::vector<double> &coefficients() const noexcept {
    return coefficients_;
  }
  /// \f$ \langle m^2\rangle_\beta=\sum_{|m|\le M}m^2c_m/\sum_{|m|\le M}c_m \f$,
  /// the second moment of the order-\f$ M \f$ sums at trivial holonomy.
  [[nodiscard]] double secondMoment() const noexcept { return secondMoment_; }
  /// \f$ \beta_V=\beta/\langle m^2\rangle_\beta \f$, with the
  /// order-\f$ M \f$ second moment.
  [[nodiscard]] double matchedWeight() const noexcept {
    return beta_ / secondMoment_;
  }

  /// \f$ W_M \f$, \f$ DW_M \f$ and \f$ D^2W_M \f$ at \p holonomy, with the
  /// rounding bound of \f$ W_M \f$ and the reported tails.
  /// @throws std::invalid_argument when \p holonomy is zero or not finite.
  [[nodiscard]] VillainSeries series(std::complex<double> holonomy) const;

  /// Whether \f$ W_M \f$ is certified nonzero at the holonomy the sums of
  /// \p point were evaluated at: the modulus of the computed sum, rounded
  /// downward, exceeds its rounding bound.
  [[nodiscard]] static bool certifiedNonzero(const VillainSeries &point);

  /// \f$ \log W_M(F) \f$ on the branch real on the unit circle, by the
  /// certified radial continuation described above.
  /// @throws std::invalid_argument when \p holonomy is zero or not finite.
  /// @throws std::domain_error when \f$ W_M \f$ is not above its bound at the
  ///   point of the unit circle on the ray of \p holonomy, when the path
  ///   meets a zero of \f$ W_M \f$ at the resolution of double precision, or
  ///   when \f$ W_M \f$ is not certified nonzero at \p holonomy.
  /// @throws std::logic_error when the imaginary part of the computed sum at
  ///   the start of the path exceeds its bound, which contradicts the
  ///   reality of \f$ W_M \f$ on the unit circle.
  [[nodiscard]] std::complex<double> logarithm(
      std::complex<double> holonomy) const;

  /// \f$ \phi(F)=-\beta_V\log W_M(F) \f$.
  [[nodiscard]] std::complex<double> potential(
      std::complex<double> holonomy) const;
  /// \f$ D\phi=-\beta_V\,DW_M/W_M \f$.
  /// @throws std::domain_error when \f$ W_M(F) \f$ is not certified nonzero.
  [[nodiscard]] std::complex<double> firstDerivative(
      std::complex<double> holonomy) const;
  /// \f$ D^2\phi=-\beta_V(D^2W_M/W_M-(DW_M/W_M)^2) \f$.
  /// @throws std::domain_error when \f$ W_M(F) \f$ is not certified nonzero.
  [[nodiscard]] std::complex<double> secondDerivative(
      std::complex<double> holonomy) const;

 private:
  /// An upper bound of
  /// \f$ \sum_{m=1}^{M}m^2c_m(\rho_+^{\,m}+\rho_-^{-m}) \f$, every
  /// operation rounded upward: the bound \f$ B \f$ on \f$ |D^2W_M| \f$ over
  /// the moduli between \p lowerModulus and \p upperModulus.
  [[nodiscard]] double secondDerivativeBound(double upperModulus,
                                             double lowerModulus) const;

  /// An upper bound of \f$ V=|DW_M|\,L+\tfrac12BL^2 \f$ at the node
  /// \p node was evaluated at, with \f$ B \f$ = \p secondBound and
  /// \f$ L \f$ = \p length, every operation rounded upward.
  [[nodiscard]] double changeBound(const VillainSeries &node,
                                   double secondBound, double length) const;

  double beta_;
  int order_;
  std::vector<double> coefficients_;
  double secondMoment_ = 1.0;
};

/// # HolonomyTruncation
///
/// The order of the Villain weight and the reported distance of its sums from
/// their infinite series over every face of the complex at the current
/// connection, as `JointAction::holonomyTruncation` reports it. Each tail is
/// the bound of `VillainSeries` divided by the sum of the moduli of the terms
/// of its own order-\f$ M \f$ sum, and the largest over the faces is kept; a
/// face at which that sum is zero in double precision has no ratio and is
/// left out of the maximum. A report: nothing reads it.
struct HolonomyTruncation {
  /// \f$ M \f$, the declared order.
  int order = 0;
  /// \f$ \max_\tau \f$ of the tail of \f$ W \f$ over
  /// \f$ \sum_{|m|\le M}c_m|\mathcal F_\tau|^m \f$.
  double relativeValueTail = 0.0;
  /// \f$ \max_\tau \f$ of the tail of \f$ DW \f$ over
  /// \f$ \sum_{|m|\le M}|m|\,c_m|\mathcal F_\tau|^m \f$.
  double relativeFirstTail = 0.0;
  /// \f$ \max_\tau \f$ of the tail of \f$ D^2W \f$ over
  /// \f$ \sum_{|m|\le M}m^2c_m|\mathcal F_\tau|^m \f$.
  double relativeSecondTail = 0.0;
};

/// # ReportedActionValue
///
/// The value of the joint action as a solver records it
/// (`JointAction::reportedValue`): the value when it can be evaluated, and
/// otherwise the reason it cannot, by name.
struct ReportedActionValue {
  /// Whether the value could be evaluated.
  bool available = true;
  /// \f$ S(z,U,\Gamma) \f$; a quiet NaN in both parts when unavailable.
  std::complex<double> value{0.0, 0.0};
  /// Why the value is unavailable (the refusal's own message); empty when it
  /// is available.
  std::string unavailable;
};

/// # CarrierDerivatives
///
/// The first derivatives of the carrier operator \f$ h_k(z,U) \f$ in the
/// coordinates a relaxation moves, one flat row-major matrix per edge in
/// `getEdgeList()` order: \f$ \partial h/\partial z_e \f$ in the squared
/// length, and the Maurer-Cartan derivative \f$ U_e\,\partial h/\partial U_e \f$
/// on the edge's stored orientation. All-zero for an edge the complex does not
/// carry; every entry empty when the complex carries no cell of the carrier
/// degree.
struct CarrierDerivatives {
  std::vector<std::vector<std::complex<double>>> lengths;
  std::vector<std::vector<std::complex<double>>> links;
};

/// # CarriedStateVariation
///
/// A first-order variation of the carried state at fixed geometry: of the
/// covariance \f$ \Gamma \f$, of the constraints' fiber projector
/// \f$ P_{\mathcal C} \f$ and of each declared band projector \f$ P_b \f$
/// (one entry per `JointActionDeclaration::momentBandProjectors`), each flat
/// row-major over the \f$ k \f$-cells. An empty entry is a zero variation.
struct CarriedStateVariation {
  std::vector<std::complex<double>> covariance;
  std::vector<std::complex<double>> momentProjector;
  std::vector<std::vector<std::complex<double>>> bandProjectors;
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
  /// equations. The dual Regge action's exact gradient and Hessian are
  /// analytic on each side of the real axis in the squared lengths and not
  /// across it: the deficit angle is taken on the principal branch and
  /// carries no Riemann-sheet label, so an arbitrarily small positive
  /// imaginary part in \f$ z_e \f$ shifts a hinge's deficit by \f$ 2\pi \f$
  /// and its contribution to the gradient by \f$ 2\pi \f$ times the hinge's
  /// dual volume. The Jacobian of a solve is assembled from the analytic
  /// derivatives on the side of the axis the point lies on (`reggeHessian`).
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

  /// \f$ \beta \f$, the heat-kernel coupling of the face-holonomy term
  /// \f$ S_{\rm hol}(U) \f$, whose coefficient is then
  /// \f$ \beta_V=\beta/\langle m^2\rangle_\beta \f$ (`VillainCharacter`); the
  /// bare connection stiffness at trivial holonomy is \f$ \beta L_1^{\rm up} \f$.
  /// Zero leaves the term out; it must be non-negative.
  double holonomyWeight = 0.0;

  /// \f$ M \f$, the order the Villain weight is summed to: the holonomy term
  /// is defined by the Laurent polynomial
  /// \f$ W_M(F)=\sum_{|m|\le M}e^{-m^2/(2\beta)}F^m \f$ (`VillainCharacter`).
  /// An integer from one to `VillainCharacter::maximumOrder`, ten, which is
  /// the default.
  int villainOrder = VillainCharacter::maximumOrder;

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

  /// The fiber the spectral constraints are imposed on: the Riesz projector
  /// \f$ P_{\mathcal C} \f$ of an isolated band, flat row-major over the
  /// \f$ k \f$-cells in the canonical order. Empty (the default), the
  /// constraints are the power sums of the whole carrier,
  /// \f$ p_j(h)=\operatorname{tr}(h^j) \f$. Declared, they are the power sums
  /// of the compressed operator
  /// \f$ h_{\mathcal C}=P_{\mathcal C}hP_{\mathcal C}|_{\operatorname{Ran}
  /// P_{\mathcal C}} \f$ of WP v17 §3.4,
  /// \f[ p_j(h_{\mathcal C})=\operatorname{tr}\bigl((P_{\mathcal C}hP_{\mathcal C})^j\bigr), \f]
  /// the \f$ j \f$-th power sum of the fiber's eigenvalues (the compression
  /// acts as zero on \f$ \ker P_{\mathcal C} \f$). Their derivatives are
  /// taken at fixed \f$ P_{\mathcal C} \f$,
  /// \f$ dp_j=j\operatorname{tr}\bigl(P_{\mathcal C}(P_{\mathcal C}hP_{\mathcal C})^{j-1}P_{\mathcal C}\,dh\bigr) \f$,
  /// which is the whole derivative when \f$ P_{\mathcal C} \f$ is a spectral
  /// projector of \f$ h \f$: the projector's own variation is off-diagonal
  /// between its range and its kernel and so contributes no trace against a
  /// power of \f$ h \f$. A self-consistent solve rebuilds the projector at
  /// every point, as it does \f$ \Gamma \f$.
  std::vector<std::complex<double>> momentProjector;

  /// \f$ s>0 \f$, the unit the power sums are measured in: the constraints
  /// are \f$ p_j(h/s)=\operatorname{tr}((h/s)^j) \f$ (or of
  /// \f$ h_{\mathcal C}/s \f$), with targets and multipliers in the same
  /// unit. The constraint \f$ p_j(h/s)=p_j^{\star}s^{-j} \f$ is the
  /// constraint \f$ p_j(h)=p_j^{\star} \f$, so the scale changes no solution
  /// set; it keeps the constraints' gradients, which grow as
  /// \f$ j|\lambda|^{j-1} \f$, commensurate with one another and with the
  /// geometric equations when several moments are pinned. A multiplier in
  /// this unit is \f$ s^j \f$ times the multiplier of the unscaled
  /// constraint. One, the default, measures in the operator's own unit.
  double momentScale = 1.0;

  /// The Riesz projectors \f$ P_b \f$ of the bands the `BandMean`
  /// constraints refer to (`SpectralMomentConstraint::band` indexes this
  /// list), each flat row-major over the \f$ k \f$-cells. Their derivatives
  /// are taken at fixed \f$ P_b \f$, the whole derivative when \f$ P_b \f$
  /// is a spectral projector of \f$ h \f$, as for `momentProjector`; a
  /// self-consistent solve rebuilds them at every point. The unit \f$ s \f$
  /// applies: the constraint is \f$ \lambda_b/s=\lambda_b^{\star}/s \f$,
  /// and a multiplier in the unit is \f$ s \f$ times the unscaled one.
  std::vector<std::vector<std::complex<double>>> momentBandProjectors;

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
/// * \f$ S_{\rm hol}(U)=\sum_\tau\phi(\mathcal F_\tau) \f$ runs over the
///   triangles \f$ \tau \f$ of the complex, with the branch-free face holonomy
///   \f$ \mathcal F_\tau=\prod_{e\subset\partial\tau}U_e^{\epsilon_{\tau e}} \f$
///   and \f$ \epsilon_{\tau e} \f$ the integer incidence of \f$ \partial_2 \f$.
///   The per-face potential is the Villain (heat-kernel) action in its
///   character form, \f$ \phi(F)=-\beta_V\log W_M(F) \f$ with the
///   character sum \f$ W_\beta(F)=\sum_{m\in\mathbb Z}e^{-m^2/(2\beta)}F^m \f$
///   summed to the declared order, \f$ W_M=\sum_{|m|\le M} \f$ with
///   \f$ M \f$ the declared `villainOrder`, and
///   \f$ \beta_V=\beta/\langle m^2\rangle_\beta \f$ (`VillainCharacter`),
///   \f$ \beta \f$ the declared `holonomyWeight`: the holonomy term of the
///   whitepaper's action at that order. It is a holomorphic function of
///   \f$ F\in\mathbb C^{*} \f$ (a Laurent polynomial) that is even under
///   \f$ F\leftrightarrow F^{-1} \f$, and \f$ \mathcal F_\tau \f$ is a Laurent
///   monomial in the links, so the term is holomorphic on
///   \f$ (\mathbb{C}^{*})^{|E|} \f$ and no argument, logarithm or modulus of a
///   link is ever taken. Evenness makes trivial holonomy stationary, and the
///   second variation there in the real link angles is \f$ \beta L_1^{\rm up} \f$
///   with the up-Laplacian \f$ L_1^{\rm up}=\partial_2\partial_2^{\mathsf T} \f$,
///   which vanishes on pure-gauge directions and is \f$ 4\beta \f$ on the
///   coexact block of the regular tetrahedron; at a quarter-turn holonomy the
///   Villain curvature is \f$ \kappa(\pm i) \f$ of `VillainCharacter`, which
///   does not vanish.
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
  ///   \f$ k \f$-cells of the complex, or when the holonomy term is declared
  ///   with a negative weight or a tolerance outside \f$ (0,1) \f$.
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

  /// Replace the carried covariance \f$ \Gamma \f$, flat row-major over the
  /// \f$ k \f$-cells. This is how a self-consistent solve rebuilds
  /// \f$ \Gamma \f$ from the carrier operator at a new geometry. Nothing else
  /// of the declaration changes: in particular the Riemann sheets of a
  /// continued Regge term stay those fixed when the instance was built, as
  /// they would not if a new instance were built at the new geometry.
  /// @throws std::invalid_argument when \p covariance is neither empty nor a
  ///   square matrix over the \f$ k \f$-cells of the complex.
  void setCovariance(std::vector<std::complex<double>> covariance);

  /// Replace the fiber the spectral constraints are imposed on
  /// (`JointActionDeclaration::momentProjector`), flat row-major over the
  /// \f$ k \f$-cells, in place and for the same reason as `setCovariance`.
  /// @throws std::invalid_argument when \p projector is neither empty nor a
  ///   square matrix over the \f$ k \f$-cells of the complex.
  void setMomentProjector(std::vector<std::complex<double>> projector);

  /// Replace the band projectors the `BandMean` constraints refer to
  /// (`JointActionDeclaration::momentBandProjectors`), in place and for the
  /// same reason as `setCovariance`; one projector per declared entry.
  /// @throws std::invalid_argument when the count differs from the declared
  ///   one or an entry is not a square matrix over the \f$ k \f$-cells.
  void setMomentBandProjectors(
      std::vector<std::vector<std::complex<double>>> projectors);

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

  /// The value of each declared constraint in the declared unit, in
  /// declaration order: the power sum \f$ p_j(h/s)=\operatorname{tr}((h/s)^j) \f$
  /// (of the compressed operator \f$ h_{\mathcal C} \f$ when a fiber is
  /// declared, `JointActionDeclaration::momentProjector`), computed by
  /// repeated multiplication so a defective or non-normal \f$ h \f$ needs no
  /// eigendecomposition and no eigenvalue ordering; or the band mean
  /// \f$ \operatorname{tr}(P_bh)/(r_bs) \f$.
  [[nodiscard]] std::vector<std::complex<double>> constraintValues() const;

  /// `constraintValues` under the name of the power-sum form; a band-mean
  /// constraint's entry is its band mean.
  [[nodiscard]] std::vector<std::complex<double>> powerSums() const;

  /// Each declared constraint's value less its target, in declaration
  /// order: the exact stationarity equation in \f$ \xi_j \f$, as a complex
  /// number whose vanishing is the constraint.
  [[nodiscard]] std::vector<std::complex<double>> momentResiduals() const;

  /// \f$ w_R\,S_{\rm Regge}(z) \f$ in the declared form and hinge set.
  [[nodiscard]] std::complex<double> reggeTerm() const;
  /// \f$ S_{\rm hol}(U) \f$, the weight included. This is the one quantity
  /// that needs \f$ \log W_M \f$, and it is evaluated on the branch real on
  /// the unit circle (`VillainCharacter::logarithm`); it throws
  /// `std::domain_error` for a face holonomy on whose ray \f$ W_M \f$ is not
  /// positive at the unit circle, or whose radial path meets a zero of
  /// \f$ W_M \f$.
  /// `HolomorphicRelaxation` reads the value only to report it: its steps and
  /// their acceptance use the stationarity residual alone.
  [[nodiscard]] std::complex<double> holonomyTerm() const;
  /// \f$ w_M\operatorname{tr}(\Gamma h(z,U)) \f$.
  [[nodiscard]] std::complex<double> matterTerm() const;
  /// \f$ \sum_j \xi_j\,(p_j(h)-p_j^{\star}) \f$.
  [[nodiscard]] std::complex<double> spectralTerm() const;
  /// The whole action \f$ S(z,U,\Gamma) \f$, the sum of the five terms above.
  [[nodiscard]] std::complex<double> value() const;

  /// The whole action as a solver reports it: `value` when it can be
  /// evaluated, and otherwise unavailable, with the reason by name. Only the
  /// holonomy term needs \f$ \log W_M \f$, and a solver reads the value only
  /// to report it (its steps and their acceptance use the stationarity
  /// residual alone), so a logarithm that has no value makes this record
  /// unavailable instead of ending the solve. That is anything `value` throws
  /// as `std::logic_error` (its `std::domain_error` for a zero of
  /// \f$ W_M \f$ on the path of the logarithm, or a \f$ W_M \f$ that is not
  /// positive above its bound on the unit circle, included).
  [[nodiscard]] ReportedActionValue reportedValue() const;

  /// \f$ \partial S/\partial z_e \f$ for every edge, in `getEdgeList()` order.
  ///
  /// Assembled from the exact analytic gradients the framework supplies: for
  /// the primal Regge form the per-hinge product rule
  /// \f$ \sum_h(\partial|h|\,\varepsilon_h+|h|\,\partial\varepsilon_h) \f$ from
  /// `mesh::Simplex::volumeGradient` and `mesh::Simplex::deficitAngleGradient`,
  /// for the dual form `simulations::ReggeSolver::actionGradientExact`, and the
  /// operator gradient `HodgeLaplacian::laplacianGradient`. No finite difference
  /// enters it, and no imaginary part is discarded: the returned number is the
  /// full complex derivative of a holomorphic function.
  [[nodiscard]] std::vector<std::complex<double>> lengthStationarity() const;

  /// \f$ U_e\,\partial S/\partial U_e \f$ for every edge, in `getEdgeList()`
  /// order and on the edge's stored orientation.
  ///
  /// The face-holonomy part is analytic in closed form,
  /// \f$ \sum_\tau \epsilon_{\tau e}\,D\phi(\mathcal F_\tau) \f$ with
  /// \f$ D=F\,d/dF \f$: \f$ -\beta_V\,DW_M/W_M \f$ per face. The operator part
  /// uses the
  /// identity \f$ U_e\,\partial/\partial U_e = -i\,\partial/\partial\varphi_e \f$
  /// for \f$ U_e=e^{i\varphi_e} \f$ together with the exact analytic
  /// `HodgeLaplacian::laplacianPhaseGradient`; that identity is a relation
  /// between derivatives and selects no branch of the logarithm.
  [[nodiscard]] std::vector<std::complex<double>> linkStationarity() const;

  /// Every term of the action with its value and its stationarity
  /// (`ActionTermGradient`), in the order regge, holonomy, matter,
  /// then one entry per declared constraint. A term whose weight is zero is
  /// listed with zero value and zero gradient, so the list has the same shape
  /// at every point. The sums over the terms are `lengthStationarity` and
  /// `linkStationarity` exactly. Costs a few stationarity evaluations, so it
  /// is for records and traces rather than for the inner loop of a solve.
  [[nodiscard]] std::vector<ActionTermGradient> termGradients() const;

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
  /// per-face second derivative is
  /// \f$ -\beta_V(D^2W_M/W_M-(DW_M/W_M)^2) \f$; no
  /// logarithm enters it.
  [[nodiscard]] std::vector<std::complex<double>> holonomyHessian() const;

  /// The order of the Villain weight and, over the current face holonomies,
  /// the reported distance of its sums from their infinite series
  /// (`HolonomyTruncation`).
  [[nodiscard]] HolonomyTruncation holonomyTruncation() const;

  /// The Hessian of the Regge term \f$ w_RS_{\rm Regge} \f$ in the squared
  /// lengths, flat row-major \f$ |E|\times|E| \f$ in `getEdgeList()` order.
  /// For the primal form it is the per-hinge product rule
  /// \f$ \sum_h\bigl(\partial_e\partial_f|h|\,\varepsilon_h
  /// +\partial_e|h|\,\partial_f\varepsilon_h+\partial_f|h|\,\partial_e\varepsilon_h
  /// +|h|\,\partial_e\partial_f\varepsilon_h\bigr) \f$ on the declared sheets
  /// (`ReggeBranch`), from `mesh::Simplex::volumeGradient`,
  /// `volumeGradientDirectionalDerivative`, the sheeted
  /// `deficitAngleGradient` and the sheeted `deficitAngleHessian`, the
  /// content root through its continued sign; for the dual form it is
  /// `simulations::ReggeSolver::actionHessianExact`. Zero when the weight is
  /// zero. Symmetric.
  [[nodiscard]] std::vector<std::complex<double>> reggeHessian() const;

  /// The Hessian of the action in the relaxed coordinates at fixed carried
  /// state: the Jacobian of (`lengthStationarity`, `linkStationarity`) with
  /// respect to the squared lengths \f$ z_e \f$ and the Maurer-Cartan
  /// increments \f$ \delta_e \f$ (\f$ U_e\mapsto U_ee^{\delta_e} \f$ on the
  /// stored orientation) at fixed \f$ \Gamma \f$, fixed projectors and fixed
  /// multipliers, flat row-major \f$ 2|E|\times2|E| \f$ in the block order
  /// lengths then links, each block in `getEdgeList()` order.
  ///
  /// It is `reggeHessian` on the length block, `holonomyHessian` on the link
  /// block, and on every block the contraction
  /// \f$ \operatorname{tr}(A\,\partial_x\partial_yh)
  /// +\sum_j\xi_j\operatorname{tr}(\partial_yX_j\,\partial_xh) \f$, with
  /// \f$ A=w_M\Gamma+\sum_j\xi_jX_j \f$ the matrix `lengthStationarity`
  /// contracts the operator's derivatives against and \f$ \partial_yX_j \f$
  /// the variation, through \f$ h \f$ at fixed projector, of a power sum's
  /// \f$ X_j=(j/s)(h/s)^{j-1} \f$ or of its compression to the declared
  /// fiber (zero for a band mean, whose \f$ X_j \f$ is its projector). The
  /// second derivatives of \f$ h \f$ are
  /// `HodgeLaplacian::laplacianGradientDirectionalDerivative`,
  /// `laplacianPhaseHessian` and `laplacianMixedDerivative`, the link
  /// coordinates carrying \f$ U\,\partial/\partial U=-i\,\partial/\partial\varphi \f$
  /// on each edge's stored orientation. Symmetric.
  ///
  /// A block whose coordinates are not asked for is left zero and the terms
  /// that live in it alone are not evaluated: with \p links false the
  /// holonomy term's Hessian, which needs \f$ DW_M/W_M \f$ and
  /// \f$ D^2W_M/W_M \f$ at every face holonomy, is not formed, so a solve
  /// of the lengths alone does not depend on it.
  /// @param lengths Whether the squared lengths are coordinates.
  /// @param links Whether the links are coordinates.
  [[nodiscard]] std::vector<std::complex<double>> actionHessian(
      bool lengths = true, bool links = true) const;

  /// The first derivatives of the carrier operator in the relaxed coordinates
  /// (`CarrierDerivatives`): `HodgeLaplacian::laplacianGradient` in each
  /// squared length and, on each edge's stored orientation,
  /// \f$ -i \f$ times `laplacianPhaseGradient` of the canonical link.
  [[nodiscard]] CarrierDerivatives carrierDerivatives() const;

  /// The variation of (`lengthStationarity`, `linkStationarity`) under a
  /// variation of the carried state at fixed geometry
  /// (`CarriedStateVariation`), concatenated in the block order of
  /// `stationarityResidual`'s first two blocks (length \f$ 2|E| \f$):
  /// \f$ \operatorname{tr}(\delta A\,\partial_xh) \f$ with
  /// \f$ \delta A=w_M\,\delta\Gamma+\sum_j\xi_j\,\delta X_j \f$, where
  /// \f$ \delta X_j \f$ is the variation of the constraint's matrix through
  /// its projector alone: \f$ \delta P_b/(r_bs) \f$ for a band mean, and for
  /// the power sums of a declared fiber the product rule in
  /// \f$ P_{\mathcal C} \f$ on
  /// \f$ (j/s)P_{\mathcal C}(P_{\mathcal C}hP_{\mathcal C}/s)^{j-1}P_{\mathcal C} \f$;
  /// zero for the power sums of the whole carrier, which no projector enters.
  /// This is the part of the Jacobian of a self-consistent solve that the
  /// rebuilt state contributes, once the state's variation along a coordinate
  /// is known (`HolomorphicRelaxation::jacobian`).
  /// @throws std::invalid_argument when a variation is neither empty nor a
  ///   square matrix over the \f$ k \f$-cells, or when the band projector
  ///   variations are neither absent nor one per declared band projector.
  [[nodiscard]] std::vector<std::complex<double>> stationarityStateVariation(
      const CarriedStateVariation &variation) const;


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
  /// constraint (of \f$ h_{\mathcal C} \f$, at fixed \f$ P_{\mathcal C} \f$,
  /// when a fiber is declared), concatenated in the same block order as
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

  /// The names of the four declared terms, in the order `value` sums them:
  /// `"regge"`, `"holonomy"`, `"matter"`, `"spectral"`.
  [[nodiscard]] static std::vector<std::string> termNames();

 private:
  /// The declared sheets of the primal Regge sum at the current geometry: per
  /// hinge, the sheets of its dihedral angles and the sign of its content root
  /// relative to the principal one.
  struct ReggeSheets;
  [[nodiscard]] ReggeSheets reggeSheets() const;

  /// Which part of the stationarity `stationarityPart` forms.
  enum class StationarityPart { All, Regge, Holonomy, Contraction };
  /// The one engine behind `lengthStationarity`, `linkStationarity` and
  /// `termGradients`: the Regge part on the lengths, the
  /// holonomy part on the links, and the carrier's contraction matrix traced
  /// against the operator's derivatives on both. `All` forms every part with
  /// the declared contraction (the matter and constraint terms together);
  /// `Contraction` forms the trace of the given flat matrix alone; the others
  /// form one geometric part alone. A null output is not formed.
  void stationarityPart(StationarityPart part,
                        const std::vector<std::complex<double>> *contraction,
                        std::vector<std::complex<double>> *lengths,
                        std::vector<std::complex<double>> *links) const;

  std::shared_ptr<Spacetime> spacetime_;
  JointActionDeclaration declaration_;
  /// \f$ z^{(0)} \f$ keyed by the edge's sorted vertex ids.
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>
      reggeStart_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_JOINTACTION_H
