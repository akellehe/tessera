// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_DRESSEDFLUCTUATION_H
#define TESSERA_COBORDISM_DRESSEDFLUCTUATION_H

#include <complex>
#include <cstddef>
#include <vector>

#include "cobordism/Certificate.h"
#include "cobordism/JointAction.h"

namespace tessera::cobordism {

/// # DressedFluctuationDeclaration
///
/// Everything that fixes which fluctuation problem a `DressedFluctuation`
/// instance is. Plain data, so an instance can be built, copied and recorded
/// without reaching any engine state.
///
/// The names follow the whitepaper's Section 7 setting. A retained fluctuation
/// is a vector \f$ f=(f_1,\dots,f_R)^{\mathsf T}\in\mathbb{C}^R \f$ of
/// displacements of the geometry or of the connection about a stationary
/// carrier, the carrier operator depends on it linearly,
/// \f$ h(f)=h_0+\sum_a f_a O_a \f$, and its own action is the quadratic form
/// \f$ S_g(f)=\tfrac12 f^{\mathsf T}Af \f$ with \f$ A=A^{\mathsf T} \f$.
struct DressedFluctuationDeclaration {
  /// \f$ n \f$, the dimension of the one-particle carrier space: the number of
  /// cells the carrier operator acts on.
  int carrierDimension = 0;

  /// \f$ h_0 \f$, the carrier operator at the stationary configuration, flat
  /// row-major \f$ n\times n \f$. It is generally complex and non-normal; it is
  /// never symmetrized and no adjoint of it is taken anywhere in this class.
  std::vector<std::complex<double>> carrier;

  /// \f$ O_a=\partial h/\partial f_a \f$ for each retained fluctuation, in the
  /// caller's declared fluctuation order; each entry is flat row-major
  /// \f$ n\times n \f$. The number of entries is \f$ R \f$, the fluctuation
  /// count.
  ///
  /// For a connection fluctuation these are
  /// `chainhodge::CovariantChainHodge::covariantOperatorPhaseDerivative`; for a
  /// length fluctuation they are
  /// `chainhodge::CovariantChainHodge::covariantOperatorDerivative`.
  std::vector<std::vector<std::complex<double>>> couplings;

  /// \f$ \partial^2 h/\partial f_a\,\partial f_b \f$, the second derivatives
  /// the diamagnetic term is built from, given for the pairs
  /// \f$ a\le b \f$ in row-major upper-triangular order:
  /// \f$ (0,0),(0,1),\dots,(0,R-1),(1,1),\dots,(R-1,R-1) \f$, so the vector
  /// holds \f$ R(R+1)/2 \f$ entries, each flat row-major \f$ n\times n \f$. The
  /// derivative is symmetric in the two indices, so the lower triangle is the
  /// transpose-free mirror of the upper one and is not declared separately.
  ///
  /// An empty vector declares that the carrier depends on the fluctuations
  /// linearly, so that every second derivative vanishes and the diamagnetic
  /// term is exactly zero. That is a statement about the declared problem, not
  /// a truncation.
  std::vector<std::vector<std::complex<double>>> secondDerivatives;

  /// \f$ A \f$, the bare stiffness of the fluctuation's own action, flat
  /// row-major \f$ R\times R \f$. The whitepaper requires it to be complex
  /// symmetric, \f$ A=A^{\mathsf T} \f$; the deviation from that is measured
  /// and reported on every read that uses it rather than assumed.
  ///
  /// An empty vector declares a zero bare stiffness, which is the case in which
  /// the whole stiffness of the fluctuation is the one the fermions induce.
  /// `effectiveAction` then has nothing to invert and refuses.
  std::vector<std::complex<double>> bareStiffness;

  /// How many modes of \f$ h_0 \f$ the quasi-free state occupies. The
  /// polarization sum runs over occupied \f$ m \f$ against empty \f$ n \f$, so
  /// a fully empty or fully filled carrier gives an identically zero
  /// paramagnetic term.
  std::size_t occupiedModes = 0;

  /// Which modes those are (see `OccupationOrder`).
  OccupationOrder occupationOrder = OccupationOrder::AscendingRealPart;

  /// \f$ \eta\ge 0 \f$, a declared finite lifetime given to every
  /// particle-hole excitation, which replaces each excitation energy
  /// \f$ \Delta_{nm} \f$ by \f$ \Delta_{nm}-i\eta \f$ in the polarization.
  ///
  /// Zero — the default — is the whitepaper's formula exactly, under which a
  /// collective mode of a Hermitian carrier is a real frequency with no width,
  /// because a finite complex carries finitely many particle-hole energies and
  /// so no continuum for the mode to leak into. A positive value is the
  /// declared broadening that turns that finite set into a band of finite
  /// lifetime, and it is the one parameter of this class that is an
  /// approximation; it is never applied unless the caller asks for it, and its
  /// value travels on every read that used it.
  double continuumBroadening = 0.0;

  /// The relative tolerance the certificates of this instance hold against, and
  /// the threshold below which a candidate collective mode's geometric
  /// component counts as zero.
  double tolerance = 1e-8;
};

/// # CollectiveMode
///
/// One pole of the dressed fluctuation propagator
/// \f$ A_{\rm eff}(\omega)^{-1} \f$: a frequency at which the dressed stiffness
/// is singular, together with the fluctuation direction that is soft there.
/// The whitepaper names these the gauge quanta of the connection fluctuation.
struct CollectiveMode {
  /// \f$ \omega \f$, the frequency at which
  /// \f$ \det A_{\rm eff}(\omega)=0 \f$. The dressed stiffness depends on
  /// \f$ \omega \f$ only through \f$ \omega^2 \f$, so the poles come in pairs
  /// \f$ \pm\omega \f$ and both members are reported.
  std::complex<double> frequency{0.0, 0.0};

  /// The null direction \f$ x \f$ of \f$ A_{\rm eff}(\omega) \f$, length
  /// \f$ R \f$ and unit Euclidean norm: the polarization of the collective
  /// mode, the combination of retained fluctuations that is soft at this
  /// frequency.
  std::vector<std::complex<double>> polarization;

  /// \f$ \Gamma=-2\operatorname{Im}\omega \f$, the radiation rate: the rate at
  /// which the mode's occupation relaxes into the particle-hole continuum,
  /// which is twice the decay rate of its amplitude under the convention
  /// \f$ e^{-i\omega t} \f$. It is zero for a real frequency, which is a mode
  /// that does not radiate; it is positive for a frequency in the lower half
  /// plane, which is a mode that does.
  double radiationRate = 0.0;

  /// \f$ \min_p |\omega-\Delta_p| \f$ over the particle-hole excitation
  /// energies \f$ \Delta_p \f$ and their negatives: how far this mode sits from
  /// the nearest bare particle-hole excitation. Quiet NaN when the state has no
  /// particle-hole pair at all.
  double continuumDistance = 0.0;

  /// Whether \f$ \operatorname{Re}\omega \f$ lies between the smallest and the
  /// largest \f$ \operatorname{Re}\Delta_p \f$: the mode is then degenerate
  /// with the particle-hole excitations rather than bound outside them.
  bool insideParticleHoleContinuum = false;

  /// \f$ \lVert A_{\rm eff}(\omega)x\rVert
  ///    /\bigl((\lVert A+D\rVert+\lVert\Pi(\omega)\rVert)\,\lVert x\rVert\bigr) \f$:
  /// the measured relative residual of the null-vector equation this mode
  /// solves, taken against the size of the two terms that cancel at the pole
  /// rather than against the dressed stiffness, which is what vanishes there.
  double residual = 0.0;

  /// The mode's certificate. The pole search is an exact algebraic
  /// reformulation evaluated in floating point, so the grade is
  /// `AlgebraicallyExact` and the residual above is the whole error.
  Certificate certificate{};
};

/// # ManyBodySpaceRead
///
/// The effective action of the exact elimination, evaluated on the
/// \f$ N \f$-particle space of a cluster.
///
/// The basis of that space is the set of \f$ N \f$-element subsets of the
/// cluster's fiber modes, in ascending lexicographic order of the sorted mode
/// indices; a basis vector is the wedge \f$ \phi_{i_1}\wedge\dots\wedge
/// \phi_{i_N} \f$ of the fiber's right modes in that order, and its algebraic
/// dual is the matching wedge of the fiber's left modes.
struct ManyBodySpaceRead {
  /// \f$ N \f$, the number of particles the space carries.
  std::size_t particles = 0;

  /// \f$ r \f$, the rank of the cluster's fiber: the number of one-particle
  /// modes the space is built over.
  std::size_t fiberRank = 0;

  /// \f$ \binom{r}{N} \f$, the dimension of the \f$ N \f$-particle space and
  /// the order of every matrix below.
  std::size_t dimension = 0;

  /// The occupation basis: each entry is one ascending \f$ N \f$-tuple of fiber
  /// mode indices, in ascending lexicographic order over the tuples.
  std::vector<std::vector<std::size_t>> basis;

  /// \f$ \tilde\psi^{\mathsf T}h_0\psi \f$ on this space, flat row-major
  /// \f$ \dim\times\dim \f$: the second quantization of the carrier operator
  /// restricted to the fiber.
  std::vector<std::complex<double>> oneBody;

  /// \f$ -\tfrac12 J^{\mathsf T}A^{-1}J
  ///    = -\tfrac12\sum_{ab}(A^{-1})_{ab}J_aJ_b \f$ on this space, flat
  /// row-major: the whole second term of the exact elimination, with the
  /// bilinear currents \f$ J_a=\tilde\psi^{\mathsf T}O_a\psi \f$ taken as
  /// operators on the fiber's Fock space.
  std::vector<std::complex<double>> quartic;

  /// The part of `quartic` that is a one-body operator,
  /// \f$ -\tfrac12\,d\Gamma\bigl(\sum_{ab}(A^{-1})_{ab}O_aO_b\bigr) \f$, flat
  /// row-major.
  ///
  /// The product of two bilinear currents is not normal ordered; reordering it
  /// gives \f$ J_aJ_b=d\Gamma(O_aO_b)+\sum_{ijkl}(O_a)_{ij}(O_b)_{kl}
  /// \varepsilon_i\varepsilon_k\iota_l\iota_j \f$, whose first term is this
  /// one-body remainder and whose second is `normalOrderedQuartic`. The split
  /// is exact and is reported because the whitepaper's genuine quartic
  /// interaction \f$ \sum V_{ijkl}\varepsilon_i\varepsilon_j\iota_l\iota_k \f$
  /// is the normal-ordered part alone.
  std::vector<std::complex<double>> inducedOneBody;

  /// `quartic` minus `inducedOneBody`, flat row-major: the strictly quartic,
  /// normal-ordered part of the eliminated interaction. It annihilates the
  /// zero- and one-particle sectors and is the term that takes the state
  /// outside the Gaussian class.
  std::vector<std::complex<double>> normalOrderedQuartic;

  /// \f$ S_{\rm eff}=\tilde\psi^{\mathsf T}h_0\psi
  ///    -\tfrac12 J^{\mathsf T}A^{-1}J \f$ on this space, flat row-major: the
  /// sum of `oneBody` and `quartic`.
  std::vector<std::complex<double>> effectiveAction;

  /// \f$ \lVert\tilde\Phi^{\mathsf T}\Phi-I_r\rVert_F \f$ of the declared
  /// cluster frames: how far the declared left frame is from being the
  /// algebraic dual of the right one. It is the structural premise the
  /// restriction to the fiber rests on.
  double framePairingDefect = 0.0;

  /// \f$ \lVert A-A^{\mathsf T}\rVert_F/\lVert A\rVert_F \f$ of the bare
  /// stiffness: how far it is from the complex symmetric form the elimination
  /// assumes.
  double stiffnessAsymmetry = 0.0;

  /// The condition estimate of the solve against \f$ A \f$.
  double stiffnessConditioning = 0.0;

  /// The certificate of the elimination: `StructureExact` given the verified
  /// premises above, with the residual of the solve against \f$ A \f$.
  Certificate certificate{};
};

/// # DressedFluctuation
///
/// The dressed fluctuation propagator of Section 7 of the whitepaper, its
/// poles, and the exact elimination of the fluctuation.
///
/// Reference: Bohm and Pines, "A collective description of electron
/// interactions: III. Coulomb interactions in a degenerate electron gas",
/// Physical Review 92, 609 (1953) — the random-phase dressing of an
/// instantaneous interaction by the polarization of the matter it couples to,
/// whose poles are the collective modes.
/// Reference: Hubbard, "Calculation of partition functions", Physical Review
/// Letters 3, 77 (1959); Stratonovich, "On a method of calculating quantum
/// distribution functions", Soviet Physics Doklady 2, 416 (1957) — the linear
/// coupling of an auxiliary field to a bilinear, whose elimination at its
/// saddle produces the quartic below.
///
/// ## The setting
///
/// A retained fluctuation \f$ f\in\mathbb{C}^R \f$ of the geometry or of the
/// connection couples linearly to the fermion bilinear currents
/// \f$ J_a=\tilde\psi^{\mathsf T}O_a\psi \f$ with
/// \f$ O_a=\partial h/\partial f_a \f$, and the joint complex action is
/// \f[
///   S(f,\tilde\psi,\psi)=\tfrac12 f^{\mathsf T}Af
///     + \tilde\psi^{\mathsf T}h_0\psi + f^{\mathsf T}J .
/// \f]
///
/// ## The dressed stiffness
///
/// In a quasi-free state with occupied modes \f$ m \f$ and empty modes
/// \f$ n \f$ of \f$ h_0 \f$, the fluctuation's stiffness is dressed to
/// \f[
///   A_{\rm eff}(\omega)=A+D-\Pi(\omega),\qquad
///   D_{ab}=\sum_m \langle m|\partial_a\partial_b h|m\rangle,\qquad
///   \Pi_{ab}(\omega)=\sum_{m,n}
///     \frac{\Delta_{nm}\bigl[(O_a)_{mn}(O_b)_{nm}
///           +(O_a)_{nm}(O_b)_{mn}\bigr]}{\Delta_{nm}^2-\omega^2},
/// \f]
/// with \f$ D \f$ the diamagnetic (seagull) term, \f$ \Pi \f$ the paramagnetic
/// polarization and \f$ \Delta_{nm}=\lambda_n-\lambda_m \f$.
///
/// The bracket is the pair of matrix elements in both orders, which is what
/// second-order perturbation theory of the mixed second derivative
/// \f$ \partial_a\partial_b\sum_m\lambda_m \f$ produces; the whitepaper writes
/// the numerator as \f$ 2\Delta_{nm}(O_a)_{mn}(O_b)_{nm} \f$, which is the same
/// number whenever the two orders agree and is the form to be symmetrized when
/// they do not. They do not agree in general here, because the carrier is
/// self-adjoint against a metric that itself moves with the fluctuation, so
/// \f$ O_a \f$ is not self-adjoint in the metric at the stationary point. Taken
/// in the symmetric form, \f$ \Pi \f$ is complex symmetric, and
/// \f$ D-\Pi(0) \f$ is exactly the Hessian of the occupied energy, which is the
/// statement the Ward identity below rests on.
///
/// ## The transpose pairing
///
/// The carrier is generically complex and non-normal, so the brackets above are
/// the complex bilinear ones and no adjoint appears. Writing the
/// eigendecomposition \f$ h_0=V\Lambda V^{-1} \f$, the right modes are the
/// columns \f$ \phi_a \f$ of \f$ V \f$ and the left modes are the rows
/// \f$ \tilde\phi_a^{\mathsf T} \f$ of \f$ V^{-1} \f$, so that
/// \f$ \tilde\Phi^{\mathsf T}\Phi=I \f$ with no conjugation. A matrix element
/// is then \f$ (O_a)_{mn}=\tilde\phi_m^{\mathsf T}O_a\phi_n=(V^{-1}O_aV)_{mn} \f$
/// and the diamagnetic expectation is
/// \f$ \tilde\phi_m^{\mathsf T}(\partial_a\partial_b h)\phi_m \f$. Under this
/// pairing \f$ D \f$ and \f$ \Pi \f$ are complex symmetric whenever the
/// declared second derivatives are, and neither is Hermitian; the Hermitian
/// specialization is recovered when the carrier is self-adjoint, where
/// \f$ V^{-1}=V^\dagger \f$.
///
/// ## The Ward identity
///
/// At \f$ \omega=0 \f$, \f$ D-\Pi(0) \f$ is the Hessian of the occupied energy
/// \f$ \sum_m\lambda_m(f) \f$. On a pure-gauge direction the fluctuation is a
/// similarity transformation of the carrier, which leaves every eigenvalue
/// where it was, so the Hessian vanishes there identically. The paramagnetic
/// term alone does not, so the identity is a check on the two terms together;
/// `wardCertificate` measures it against declared pure-gauge directions.
///
/// ## The poles
///
/// The poles of \f$ A_{\rm eff}(\omega)^{-1} \f$ are the collective modes: the
/// fluctuation hybridized with the particle-hole pairs of the composite, which
/// is what the whitepaper calls the gauge quanta. `collectiveModes` finds them
/// exactly rather than by sampling \f$ \omega \f$: writing
/// \f$ \frac{2\Delta_p}{\Delta_p^2-\omega^2}
///     =\frac{1}{\Delta_p-\omega}+\frac{1}{\Delta_p+\omega} \f$
/// and introducing one amplitude per particle-hole pair for each of the two
/// partial fractions turns \f$ \det A_{\rm eff}(\omega)=0 \f$ into a linear
/// matrix pencil of order \f$ R+2P \f$ in \f$ \omega \f$, with \f$ P \f$ the
/// number of particle-hole pairs, whose finite eigenvalues are exactly the
/// zeros of \f$ \det A_{\rm eff} \f$ together with the energies of any
/// particle-hole pair that does not couple. The latter are separated by their
/// vanishing geometric component and reported nowhere; the former carry the
/// measured residual of their own null-vector equation. No sampling grid, no
/// square root and no ordering convention enters.
///
/// ## The exact elimination
///
/// Stationarity in \f$ f \f$ gives \f$ f=-A^{-1}J \f$, and exact substitution
/// gives
/// \f[
///   S_{\rm eff}=\tilde\psi^{\mathsf T}h_0\psi-\tfrac12 J^{\mathsf T}A^{-1}J .
/// \f]
/// `effectiveAction` evaluates both terms on the \f$ N \f$-particle space of a
/// declared cluster fiber, with \f$ N=3 \f$ the three-particle space of the
/// whitepaper's baryon. The interaction stays factored through the
/// \f$ R\times R \f$ response \f$ A^{-1} \f$ — one factorization of \f$ A \f$
/// applied to the \f$ R \f$ current matrices — so no dense four-index tensor is
/// ever formed.
class DressedFluctuation {
 public:
  /// The largest \f$ N \f$-particle dimension `effectiveAction` will
  /// materialize before refusing. The matrices it builds are dense and there
  /// are four of them, so the cap is on the dimension rather than on the fiber
  /// rank.
  static constexpr std::size_t kDefaultManyBodyDimensionCap = 4096;

  /// Build the dressed fluctuation problem.
  ///
  /// The eigendecomposition of the carrier is taken once here, so every read
  /// below is a cheap assembly over the stored modes.
  ///
  /// @param declaration The carrier, the couplings, the second derivatives, the
  ///   bare stiffness and the occupation rule.
  /// @throws std::invalid_argument when the carrier dimension is not positive,
  ///   when the carrier or any declared matrix is not
  ///   \f$ n\times n \f$, when the second derivatives are present but are not
  ///   \f$ R(R+1)/2 \f$ of them, when the bare stiffness is present but is not
  ///   \f$ R\times R \f$, when more modes are declared occupied than the
  ///   carrier has, when the declared broadening is negative, or when the
  ///   carrier has no eigenbasis, for which no occupied/empty splitting of the
  ///   modes exists.
  explicit DressedFluctuation(DressedFluctuationDeclaration declaration);

  /// The declaration this instance was built from.
  [[nodiscard]] const DressedFluctuationDeclaration &declaration()
      const noexcept {
    return declaration_;
  }

  /// \f$ n \f$, the dimension of the one-particle carrier space.
  [[nodiscard]] std::size_t carrierDimension() const noexcept;

  /// \f$ R \f$, the number of retained fluctuations.
  [[nodiscard]] std::size_t fluctuationCount() const noexcept;

  /// \f$ P \f$, the number of particle-hole pairs: the occupied mode count
  /// times the empty mode count.
  [[nodiscard]] std::size_t particleHolePairCount() const noexcept;

  /// The eigenvalues of \f$ h_0 \f$ in the declared occupation order, so that
  /// the first `occupiedModes` entries are the occupied ones.
  [[nodiscard]] std::vector<std::complex<double>> carrierEigenvalues() const;

  /// The particle-hole excitation energies
  /// \f$ \Delta_{nm}=\lambda_n-\lambda_m \f$, one per pair of an occupied
  /// \f$ m \f$ with an empty \f$ n \f$, in the pair order occupied-major then
  /// empty-minor. The declared broadening is not applied to these; they are the
  /// bare energies.
  [[nodiscard]] std::vector<std::complex<double>> particleHoleEnergies() const;

  /// \f$ (V^{-1}O_aV) \f$, the current matrix of the \p index-th fluctuation in
  /// the carrier's mode basis, flat row-major \f$ n\times n \f$ in the declared
  /// occupation order. Its \f$ (m,n) \f$ entry is the transpose-paired matrix
  /// element \f$ \tilde\phi_m^{\mathsf T}O_a\phi_n \f$.
  /// @throws std::out_of_range when \p index names no declared fluctuation.
  [[nodiscard]] std::vector<std::complex<double>> modeCurrents(
      std::size_t index) const;

  /// \f$ D_{ab}=\sum_m\tilde\phi_m^{\mathsf T}(\partial_a\partial_b h)\phi_m \f$,
  /// the diamagnetic term, flat row-major \f$ R\times R \f$. Identically zero
  /// when no second derivatives are declared.
  [[nodiscard]] std::vector<std::complex<double>> diamagnetic() const;

  /// \f$ \Pi(\omega) \f$, the paramagnetic polarization, flat row-major
  /// \f$ R\times R \f$.
  /// @throws std::domain_error when \p frequency coincides with a
  ///   particle-hole energy to within the declared tolerance, where the
  ///   polarization has a pole and no finite value to report.
  [[nodiscard]] std::vector<std::complex<double>> paramagnetic(
      std::complex<double> frequency = {0.0, 0.0}) const;

  /// \f$ A_{\rm eff}(\omega)=A+D-\Pi(\omega) \f$, flat row-major
  /// \f$ R\times R \f$.
  /// @throws as `paramagnetic`.
  [[nodiscard]] std::vector<std::complex<double>> dressedStiffness(
      std::complex<double> frequency = {0.0, 0.0}) const;

  /// \f$ D-\Pi(0) \f$, the induced stiffness, flat row-major
  /// \f$ R\times R \f$: the whole fermion contribution to the dressed
  /// stiffness, with the bare term left out so that the induced and the bare
  /// stiffness can be compared. It is the Hessian of the occupied energy
  /// \f$ \sum_m\lambda_m(f) \f$ in the retained fluctuations.
  [[nodiscard]] std::vector<std::complex<double>> inducedStiffness() const;

  /// The Ward identity: the induced stiffness \f$ D-\Pi(0) \f$ vanishes on
  /// every pure-gauge direction, because along one the carrier moves by a
  /// similarity transformation and its eigenvalues do not move at all.
  ///
  /// @param gaugeDirections The pure-gauge directions, each a vector of length
  ///   \f$ R \f$. For a connection fluctuation on the edges of a complex these
  ///   are the columns of the coboundary
  ///   \f$ \delta\varphi_{xy}=\chi_y-\chi_x \f$ of the vertex functions
  ///   \f$ \chi \f$, which is \f$ \partial_1^{\mathsf T} \f$.
  /// @return A certificate whose residual is
  ///   \f$ \max_g\lVert (D-\Pi(0))g\rVert
  ///       /(\lVert D-\Pi(0)\rVert\,\lVert g\rVert) \f$
  ///   over the declared directions, graded `AlgebraicallyExact` because the
  ///   identity is an exact consequence of similarity invariance and the only
  ///   error is rounding. A run that declares no direction reports an
  ///   unmeasured residual, which never holds.
  /// @throws std::invalid_argument when a direction has the wrong length.
  [[nodiscard]] Certificate wardCertificate(
      const std::vector<std::vector<std::complex<double>>> &gaugeDirections)
      const;

  /// The residual of the Ward identity on one declared direction, as the plain
  /// number `wardCertificate` maximizes.
  /// @throws std::invalid_argument when the direction has the wrong length.
  [[nodiscard]] double wardResidual(
      const std::vector<std::complex<double>> &gaugeDirection) const;

  /// The poles of \f$ A_{\rm eff}(\omega)^{-1} \f$: every frequency at which
  /// the dressed stiffness is singular, with its polarization, its radiation
  /// rate and its position relative to the particle-hole excitations.
  ///
  /// The modes come back ascending by
  /// \f$ (\operatorname{Re}\omega,\operatorname{Im}\omega) \f$. A candidate
  /// whose geometric component is smaller than the declared tolerance is an
  /// uncoupled particle-hole excitation rather than a pole of the propagator
  /// and is not reported; so is a candidate whose measured null-vector residual
  /// exceeds the declared tolerance, which is the reading of an eigenvalue the
  /// linearization produced but the dressed stiffness does not confirm.
  [[nodiscard]] std::vector<CollectiveMode> collectiveModes() const;

  /// The effective action of the exact elimination on the \p particles-particle
  /// space of a cluster.
  ///
  /// @param clusterFrame \f$ \Phi_C \f$, the cluster fiber's right frame, flat
  ///   row-major \f$ n\times r \f$: its \f$ r \f$ columns span the fiber inside
  ///   the carrier space. An empty frame declares the whole carrier space, so
  ///   that \f$ r=n \f$ and both frames are the identity.
  /// @param clusterDualFrame \f$ \tilde\Phi_C^{\mathsf T} \f$, the fiber's left
  ///   frame, flat row-major \f$ r\times n \f$, the algebraic dual of
  ///   \p clusterFrame with \f$ \tilde\Phi_C^{\mathsf T}\Phi_C=I_r \f$. Empty
  ///   under the same convention.
  /// @param particles \f$ N \f$, the number of particles. Three is the
  ///   whitepaper's three-particle space of a baryon.
  /// @param dimensionCap The largest \f$ \binom{r}{N} \f$ this call will
  ///   materialize.
  /// @throws std::invalid_argument when a frame has the wrong size, when the
  ///   two frames disagree on \f$ r \f$, when \p particles exceeds \f$ r \f$,
  ///   or when no bare stiffness is declared, which leaves nothing to invert;
  ///   std::length_error when \f$ \binom{r}{N} \f$ exceeds \p dimensionCap.
  [[nodiscard]] ManyBodySpaceRead effectiveAction(
      const std::vector<std::complex<double>> &clusterFrame,
      const std::vector<std::complex<double>> &clusterDualFrame,
      std::size_t particles = 3,
      std::size_t dimensionCap = kDefaultManyBodyDimensionCap) const;

 private:
  struct Modes;

  DressedFluctuationDeclaration declaration_;
  std::size_t dimension_ = 0;
  std::size_t fluctuations_ = 0;
  // The carrier's right modes in the declared occupation order, the matching
  // left modes as the rows of the inverse, and the ordered eigenvalues.
  std::vector<std::complex<double>> rightModes_;   // V, n x n row-major
  std::vector<std::complex<double>> leftModes_;    // V^{-1}, n x n row-major
  std::vector<std::complex<double>> eigenvalues_;  // ordered
  // (V^{-1} O_a V) for every declared fluctuation, in declaration order.
  std::vector<std::vector<std::complex<double>>> modeCurrents_;
  // The assembled diamagnetic term, built once.
  std::vector<std::complex<double>> diamagnetic_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_DRESSEDFLUCTUATION_H
