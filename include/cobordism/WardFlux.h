// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_WARDFLUX_H
#define TESSERA_COBORDISM_WARDFLUX_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cobordism/JointAction.h"
#include "observables/ClusterLineage.h"

namespace tessera::cobordism {

/// # WardFluxConfig
///
/// Every threshold of the Ward-flux read, echoed on each read so a stored
/// result carries the configuration that produced it.
struct WardFluxConfig {
  /// \f$ |(\partial j)_x| \f$ at or below this counts as a vanishing
  /// divergence. It is an absolute number and is compared against the
  /// divergence measured at the interior vertices of the cobordism, and
  /// against the flux difference of two homologous cuts.
  double divergenceTolerance = 1e-9;
  /// \f$ |\varphi_j(\Sigma) - n| \f$ at or below this, for the nearest integer
  /// \f$ n \f$, lets the flux be reported as the integer quark number
  /// \f$ N_q \f$. Above it the flux is reported as the complex number it is
  /// and no integer is claimed.
  double integralityTolerance = 1e-6;
  /// \f$ |\operatorname{Im}\varphi_j(\Sigma)| \f$ must be at or below this for
  /// the flux to be read as an integer. The flux is a complex number by
  /// construction and its imaginary part is never discarded silently.
  double imaginaryTolerance = 1e-9;
  /// \f$ |\varphi_j(\Sigma) - Q_{\rm in}| \f$ at or below this counts as the
  /// flux agreeing with the charge the incoming state places on
  /// \f$ \partial_{\rm in}W \f$. Above it the disagreement is named.
  double chargeTolerance = 1e-6;
};

/// # WardFluxRead
///
/// The flux \f$ \varphi_j(\Sigma)=\langle j,\Sigma\rangle \in \mathbb{C} \f$ of
/// the complex Ward current through one cooriented cut \f$ \Sigma \f$ of the
/// interaction cobordism \f$ W \f$, with every certificate the whitepaper
/// attaches to it.
///
/// Vocabulary used below. \f$ u \f$ is the cut's 0-cochain
/// (`observables::CoorientedCut::side`): 0 on the incoming side, 1 on the
/// outgoing side. \f$ \partial_{\rm in}W \f$ is the first level of the history
/// and \f$ \partial_{\rm out}W \f$ its last level. An interior vertex is a
/// vertex of \f$ W \f$ on neither. \f$ (\partial j)_x \f$ is the discrete
/// divergence of the current at vertex \f$ x \f$, with the chain complex's
/// boundary convention \f$ \partial[x<y]=[y]-[x] \f$.
struct WardFluxRead {
  /// The canonical \f$ C_1(W) \f$ indices of the edges the cut crosses,
  /// copied from `observables::CoorientedCut::crossingEdges`.
  std::vector<int> crossingEdges{};
  /// The coorientation \f$ c_e=u(b)-u(a)\in\{+1,-1\} \f$ of each crossing
  /// edge \f$ e=(a<b) \f$, parallel to `crossingEdges`.
  std::vector<int> crossingSigns{};
  /// The Ward current \f$ j_e \f$ on each crossing edge, on the canonical
  /// ascending-vertex orientation and parallel to `crossingEdges`.
  std::vector<std::complex<double>> crossingCurrent{};

  /// \f$ \varphi_j(\Sigma)=\langle j,\Sigma\rangle=\sum_{e}c_e\,j_e \f$ over
  /// the crossing edges. Complex, and never projected onto a real part.
  std::complex<double> flux{0.0, 0.0};

  /// \f$ \sum_{x:\,u(x)=0}(\partial j)_x \f$, the divergence summed over the
  /// cut's incoming side. The discrete divergence theorem makes it exactly
  /// \f$ -\varphi_j(\Sigma) \f$; it is reported so that the identity is
  /// measured rather than assumed.
  std::complex<double> incomingSideDivergence{0.0, 0.0};
  /// \f$ |\varphi_j(\Sigma)+\sum_{u(x)=0}(\partial j)_x| \f$, the residual of
  /// the divergence theorem. Zero to rounding on every read.
  double divergenceTheoremResidual = std::numeric_limits<double>::quiet_NaN();

  /// \f$ -\sum_{x\in\partial_{\rm in}W}(\partial j)_x \f$: the charge the
  /// current itself carries in through the incoming boundary. The flux equals
  /// this number plus the divergence carried by the interior vertices on the
  /// incoming side, so on a current with no bulk source the two agree.
  std::complex<double> incomingBoundaryDivergence{0.0, 0.0};

  /// \f$ \max_x |(\partial j)_x| \f$ over the interior vertices of \f$ W \f$:
  /// the discrete Ward identity \f$ \partial j=0 \f$ in the bulk, measured.
  /// NaN when \f$ W \f$ has no interior vertex, as a history of two levels
  /// does not.
  double bulkDivergenceMax = std::numeric_limits<double>::quiet_NaN();
  /// How many interior vertices `bulkDivergenceMax` was taken over.
  std::size_t bulkVertices = 0;
  /// The vertex at which `bulkDivergenceMax` was attained; empty when there
  /// is no interior vertex. A source in the bulk is named by its vertex rather
  /// than reported as an anonymous norm.
  std::optional<std::uint64_t> bulkDivergenceVertex{};

  /// \f$ Q_{\rm in}=\sum_{\sigma\subset\partial_{\rm in}W}\Gamma_{\sigma\sigma}
  /// \f$: the fermion number the declared covariance places on the carrier
  /// cells lying wholly in the incoming boundary. This is the incoming
  /// state's charge, read from the covariance and independently of the
  /// current, so that the statement "the flux is the charge carried in
  /// through \f$ \partial_{\rm in}W \f$" is a measurement and not a
  /// definition. Empty when the action declares no covariance.
  std::optional<std::complex<double>> incomingBoundaryCharge{};
  /// How many carrier cells lie wholly in \f$ \partial_{\rm in}W \f$.
  std::size_t incomingBoundaryCells = 0;
  /// \f$ Q_{\rm out}=\sum_{\sigma\subset\partial_{\rm out}W}
  /// \Gamma_{\sigma\sigma} \f$, the same reading on the outgoing boundary.
  /// Empty when the action declares no covariance.
  std::optional<std::complex<double>> outgoingBoundaryCharge{};
  /// How many carrier cells lie wholly in \f$ \partial_{\rm out}W \f$.
  std::size_t outgoingBoundaryCells = 0;
  /// \f$ |\varphi_j(\Sigma)-Q_{\rm in}| \f$, the defect of the identity "the
  /// flux equals the charge carried in through the incoming boundary". NaN
  /// when no covariance is declared.
  double boundaryChargeResidual = std::numeric_limits<double>::quiet_NaN();

  /// \f$ N_q \f$: the integer nearest
  /// \f$ \operatorname{Re}\varphi_j(\Sigma) \f$, present only when the flux is
  /// integral within `integralityTolerance` and its imaginary part is within
  /// `imaginaryTolerance`. Empty means unknown, never zero.
  std::optional<long long> quarkNumber{};
  /// \f$ |\varphi_j(\Sigma)-N_q| \f$ against the nearest integer, reported
  /// whether or not the integer was accepted.
  double quarkNumberDefect = std::numeric_limits<double>::quiet_NaN();
  /// \f$ B(\Sigma)=N_q/3 \f$. The factor one third is the whitepaper's one
  /// explicit physical calibration and not a topological theorem. Empty
  /// exactly when `quarkNumber` is empty.
  std::optional<double> baryonNumber{};

  /// Whether the cut separates \f$ \partial_{\rm in}W \f$ from
  /// \f$ \partial_{\rm out}W \f$, copied from
  /// `observables::CoorientedCut::separates`.
  bool cutSeparates = false;
  /// Named failures: "cut-does-not-separate", "empty-cut",
  /// "no-interior-vertex", "bulk-source", "flux-is-not-the-incoming-charge",
  /// "nonintegral-flux", "complex-flux".
  std::vector<std::string> failedCertificates{};
};

/// # WardSlabRead
///
/// Two cuts of one homology class compared through the slab between them.
///
/// For cuts \f$ u \f$ and \f$ u' \f$ the divergence theorem gives exactly
/// \f$ \varphi_j(u)-\varphi_j(u')=\sum_x\,(u(x)-u'(x))\,(\partial j)_x \f$:
/// the flux difference is the signed divergence the slab carries. Two
/// separating cuts agree on \f$ \partial W \f$, so the slab consists of
/// interior vertices only and the difference vanishes on a current with no
/// bulk source.
struct WardSlabRead {
  /// The position of the first cut in the list supplied.
  std::size_t first = 0;
  /// The position of the second cut in the list supplied.
  std::size_t second = 0;
  /// The vertices on which the two cuts disagree: the slab.
  std::vector<std::uint64_t> slabVertices{};
  /// \f$ \varphi_j(u)-\varphi_j(u') \f$.
  std::complex<double> fluxDifference{0.0, 0.0};
  /// \f$ \sum_x(u(x)-u'(x))(\partial j)_x \f$, the signed slab divergence.
  std::complex<double> slabDivergence{0.0, 0.0};
  /// \f$ |\varphi_j(u)-\varphi_j(u')-\sum_x(u(x)-u'(x))(\partial j)_x| \f$,
  /// the residual of the slab identity. Zero to rounding.
  double slabIdentityResidual = std::numeric_limits<double>::quiet_NaN();
};

/// # WardHomologyRead
///
/// Several cuts of one cobordism read together: the whitepaper's requirement
/// that homologous cuts give the same current flux whenever no source lies in
/// the slab between them.
struct WardHomologyRead {
  /// One read per declared cut, in the order the cuts were supplied.
  std::vector<WardFluxRead> cuts{};
  /// One comparison per unordered pair of cuts, in lexicographic order of the
  /// pair.
  std::vector<WardSlabRead> slabs{};
  /// \f$ \max|\varphi_j(\Sigma_i)-\varphi_j(\Sigma_k)| \f$ over the pairs. NaN
  /// for fewer than two cuts.
  double maxFluxDeviation = std::numeric_limits<double>::quiet_NaN();
  /// \f$ \max\sum_{x\in{\rm slab}}|(\partial j)_x| \f$ over the pairs: the
  /// source content the invariance statement excludes. A nonzero deviation
  /// with a nonzero slab divergence is a source, not a broken identity. NaN
  /// for fewer than two cuts.
  double maxSlabDivergence = std::numeric_limits<double>::quiet_NaN();
  /// Whether every pair agreed to `divergenceTolerance`. False for fewer than
  /// two cuts, since no comparison was made.
  bool invariant = false;
};

/// # IntrinsicResponseConfig
///
/// The declared parameters of the intrinsic spectral response.
struct IntrinsicResponseConfig {
  /// The current the left restriction \f$ \tilde\rho_L \f$ is cut from, in
  /// canonical degree-one cell order and on each cell's canonical orientation,
  /// exactly like `JointAction::canonicalWardCurrent`.
  ///
  /// The whitepaper names a right restriction \f$ \rho_R \f$ and a left
  /// restriction \f$ \tilde\rho_L \f$ of "the complex Ward current" without
  /// saying how the two differ, and the choice changes \f$ \Upsilon_Q \f$, so
  /// it is supplied rather than assumed. An empty vector means
  /// \f$ \tilde\rho_L=\rho_R \f$: one current paired with itself through the
  /// transpose, which is the framework's pairing rule everywhere else — every
  /// pairing is the transpose and no conjugation enters. A caller that wants a
  /// genuinely distinct left current builds the action it belongs to — the
  /// same declaration with the transposed covariance, or with the inverse
  /// connection — and passes that action's `canonicalWardCurrent` here.
  std::vector<std::complex<double>> leftCurrent{};
  /// Two eigenvalues of the slice operator within this absolute distance are
  /// one degenerate band and share one Riesz projector, so no eigenvector of a
  /// degenerate band is ever ordered or selected individually.
  double degeneracyTolerance = 1e-9;
  /// \f$ |\lambda-\lambda_a| \f$ must exceed this for a sample point to be
  /// evaluated; at or below it the sample sits on a pole and the value is
  /// reported as unavailable rather than as a large finite number.
  double poleTolerance = 1e-12;
  /// Quadrature nodes of the Riesz contour each band's residue is read on. The
  /// trapezoid rule on a circle is spectrally accurate for a function
  /// holomorphic in an annulus around it, which the resolvent form is once the
  /// contour separates the band from every other band.
  int contourNodes = 64;
};

/// # IntrinsicResponseRead
///
/// The intrinsic spectral response
/// \f$ \Upsilon_Q(\lambda)
///   =\tilde\rho_L^{\mathsf T}(L_\Sigma-\lambda I)^{-1}\rho_R \f$
/// of Section 13.5, read on one cooriented cut.
///
/// It is not an electromagnetic form factor and it is not a structure factor:
/// \f$ \lambda \f$ is an eigenvalue of the slice operator and is never
/// relabelled as a momentum transfer or a momentum transfer squared.
struct IntrinsicResponseRead {
  /// The cut's crossing edges as canonical \f$ C_1(W) \f$ indices, in the
  /// order of `observables::CoorientedCut::crossingEdges`. These are the
  /// slice's coordinates: \f$ L_\Sigma \f$, \f$ \rho_R \f$ and
  /// \f$ \tilde\rho_L \f$ are all indexed by them.
  std::vector<int> sliceCells{};
  /// \f$ \rho_R \f$: the Ward current restricted to the cut, on the canonical
  /// orientation of each cut edge and carrying the cut's coorientation, so
  /// that \f$ \sum_e (\rho_R)_e \f$ is the flux.
  std::vector<std::complex<double>> rhoRight{};
  /// \f$ \tilde\rho_L \f$: the left restriction, cut from
  /// `IntrinsicResponseConfig::leftCurrent`.
  std::vector<std::complex<double>> rhoLeft{};
  /// \f$ L_\Sigma \f$, the slice operator: the principal submatrix of the
  /// degree-one metric Hodge operator \f$ h_1(z,U) \f$ on the crossing edges,
  /// flat row-major. It is generally non-normal and is never symmetrized.
  std::vector<std::complex<double>> sliceOperator{};

  /// The distinct eigenvalues of \f$ L_\Sigma \f$, grouped at
  /// `degeneracyTolerance` and each reported as the mean of its band. These
  /// are the poles of \f$ \Upsilon_Q \f$.
  std::vector<std::complex<double>> poles{};
  /// The algebraic multiplicity of each pole, parallel to `poles`.
  std::vector<std::size_t> poleMultiplicity{};
  /// The residue \f$ -\tilde\rho_L^{\mathsf T}P_a\rho_R \f$ of
  /// \f$ \Upsilon_Q \f$ at each pole, parallel to `poles`, with \f$ P_a \f$
  /// the Riesz projector of the whole degenerate band. The sign is the one the
  /// definition forces: near \f$ \lambda_a \f$,
  /// \f$ \Upsilon_Q(\lambda)\simeq
  ///   \tilde\rho_L^{\mathsf T}P_a\rho_R/(\lambda_a-\lambda) \f$, so the
  /// residue in \f$ \lambda \f$ is minus that number.
  std::vector<std::complex<double>> residues{};

  /// The sample points \f$ \lambda \f$ the response was evaluated at.
  std::vector<std::complex<double>> samples{};
  /// \f$ \Upsilon_Q(\lambda) \f$ at each sample, parallel to `samples`. A
  /// sample on a pole carries NaN.
  std::vector<std::complex<double>> response{};
  /// \f$ d\Upsilon_Q/d\lambda
  ///     = \tilde\rho_L^{\mathsf T}(L_\Sigma-\lambda I)^{-2}\rho_R \f$ at each
  /// sample, the intrinsic radius response: the analytic slope of the
  /// response, taken exactly from the resolvent and never by a finite
  /// difference. No square root and no real projection is taken of it.
  std::vector<std::complex<double>> slope{};

  /// Named failures: "empty-cut", "singular-slice-operator",
  /// "bands-not-separated", "sample-on-a-pole".
  std::vector<std::string> failedCertificates{};
};

/// # WardFlux
///
/// The flux of the complex Ward current through a cooriented cut of the
/// interaction cobordism (Section 13.4) and the intrinsic spectral response
/// that cut carries (Section 13.5).
///
/// ## The setting
///
/// Everything is read on the interaction cobordism \f$ W \f$ of a history of
/// levels, `observables::InteractionCobordism`, with oriented boundary
/// \f$ \partial W=\overline{\partial_{\rm in}W}\sqcup\partial_{\rm out}W \f$,
/// and on the cooriented cuts \f$ \Sigma \f$ of `observables::CoorientedCut`
/// that separate \f$ \partial_{\rm in}W \f$ from \f$ \partial_{\rm out}W \f$.
/// These are the cuts a quark lineage \f$ c_Q \f$ is intersected with in
/// `observables::ClusterLineage`, so the flux and the lineage number are read
/// on the same cuts. The joint action must be declared over a triangulation
/// of \f$ W \f$ itself: its chain complex must carry exactly the vertices and
/// the edges of \f$ W \f$, and a read on any other complex is refused.
///
/// The current is the link stationarity vector of the joint action,
/// \f$ j_{xy}=U_{xy}\,\partial S/\partial U_{xy} \f$, which `JointAction`
/// supplies as `canonicalWardCurrent`. Nothing here re-derives it and nothing
/// here supplies a field of its own.
///
/// ## The flux
///
/// With \f$ u \f$ the cut's 0-cochain, \f$ \Sigma \f$ is dual to
/// \f$ \delta u \f$ and
/// \f[
///   \varphi_j(\Sigma)=\langle j,\Sigma\rangle=\langle\delta u, j\rangle
///   =\sum_{e=(a<b)}j_e\,\bigl(u(b)-u(a)\bigr),
/// \f]
/// the sum running over the crossing edges alone, each carrying its
/// coorientation sign. Because the boundary convention is
/// \f$ \partial[x<y]=[y]-[x] \f$, the discrete divergence theorem reads
/// \f$ \varphi_j(\Sigma)=-\sum_{u(x)=0}(\partial j)_x \f$, and both sides are
/// reported.
///
/// ## The relative reading
///
/// The whitepaper states \f$ \partial j=0 \f$ in the bulk on the matter
/// equations of motion, so \f$ \varphi_j \f$ is unchanged between homologous
/// cuts, and that the flux is the fermion number \f$ N_q \f$. On \f$ W \f$
/// this is a relative statement: the incoming side of every separating cut
/// consists of \f$ \partial_{\rm in}W \f$ and interior vertices, so
/// \f[
///   \varphi_j(\Sigma)=-\sum_{x\in\partial_{\rm in}W}(\partial j)_x
///                     -\sum_{x\ {\rm interior},\,u(x)=0}(\partial j)_x ,
/// \f]
/// and with no bulk source the flux is the charge the current carries in
/// through the incoming boundary. The read reports each piece: the flux; the
/// divergence at every interior vertex, whose maximum is the Ward identity
/// measured and is named as a source when it does not vanish; the charge the
/// current brings in through \f$ \partial_{\rm in}W \f$; and, read
/// independently from the declared covariance, the fermion number the
/// incoming state places on \f$ \partial_{\rm in}W \f$,
/// \f$ Q_{\rm in}=\sum_{\sigma\subset\partial_{\rm in}W}\Gamma_{\sigma\sigma}
/// \f$, together with its difference from the flux. The equality of the flux
/// with \f$ Q_{\rm in} \f$ is a measurement of this framework and not a
/// definition inside it; when it fails the failure is named and the numbers
/// are reported as they come out.
///
/// The flux is not electric charge. Every edge mode carries charge one under
/// the \f$ \mathbb{C}^{*} \f$ group, so the flux counts fermions; a
/// flavor-dependent electric charge is not a gauge charge of the declared
/// fields and carries no Ward current, which the whitepaper records as a limit
/// of the present field content. Nothing here reports one.
///
/// ## The intrinsic response
///
/// When stable translation generators have not emerged there is no momentum
/// transfer and no form factor, and a slice eigenvalue is not relabelled as
/// one. The response is instead
/// \f$ \Upsilon_Q(\lambda)
///   =\tilde\rho_L^{\mathsf T}(L_\Sigma-\lambda I)^{-1}\rho_R \f$
/// with \f$ \rho_R \f$ and \f$ \tilde\rho_L \f$ the right and left
/// restrictions of the current to the cut's crossing edges and
/// \f$ L_\Sigma \f$ the slice operator. Its poles are the eigenvalues of
/// \f$ L_\Sigma \f$, its residues are taken through Riesz projectors so a
/// degenerate band is handled whole, and its slope in \f$ \lambda \f$ is the
/// analytic second power of the resolvent rather than a finite difference.
///
/// ## Background removal
///
/// Section 13.5 forms
/// \f$ \Delta O=O_{\rm state}-O_{\rm matched\ \partial_{\rm in}W} \f$
/// for every current and every response before any boundary probability, so
/// that complex background and excitation terms may cancel. `difference`
/// performs that subtraction coherently on two flux reads, and never on their
/// moduli.
///
/// ## Boundaries
///
/// Read-only: it reads an action, a cobordism and a declared cut, never calls
/// a solver, never writes the geometry, and never enters an emergence
/// objective. An unmeasured value is NaN or an empty optional with the reason
/// named, never zero.
class WardFlux {
 public:
  /// The flux of the action's Ward current through one cut of \f$ W \f$.
  ///
  /// @param action The joint action whose link stationarity is the current,
  ///   declared over a triangulation of \f$ W \f$.
  /// @param cobordism The interaction cobordism \f$ W \f$.
  /// @param cut A cooriented cut of \f$ W \f$, from
  ///   `observables::ClusterLineage::levelCut` or `cutFromSides`.
  /// @param cfg The declared thresholds.
  /// @throws std::invalid_argument when the action carries no spacetime, when
  ///   the action's complex does not carry exactly the vertices and edges of
  ///   \f$ W \f$, or when the cut does not assign one side per vertex of
  ///   \f$ W \f$.
  [[nodiscard]] static WardFluxRead flux(
      const JointAction &action,
      const observables::InteractionCobordism &cobordism,
      const observables::CoorientedCut &cut, const WardFluxConfig &cfg = {});

  /// Several cuts of one cobordism read together, with the flux difference
  /// and the signed slab divergence of every pair.
  /// @throws std::invalid_argument under the same conditions as `flux`.
  [[nodiscard]] static WardHomologyRead homologousFluxes(
      const JointAction &action,
      const observables::InteractionCobordism &cobordism,
      const std::vector<observables::CoorientedCut> &cuts,
      const WardFluxConfig &cfg = {});

  /// The coherent background removal
  /// \f$ \Delta\varphi=\varphi_{\rm state}-\varphi_{\rm matched} \f$ of two
  /// flux reads: the complex difference of the fluxes, of the divergences and
  /// of the boundary charges, with no modulus taken on either side. The
  /// returned read carries the state read's cut and the differenced numbers.
  /// @throws std::invalid_argument when the two reads were taken on different
  ///   crossing edges, since a difference between different cuts is not a
  ///   background removal.
  [[nodiscard]] static WardFluxRead difference(const WardFluxRead &state,
                                               const WardFluxRead &matched,
                                               const WardFluxConfig &cfg = {});

  /// The intrinsic spectral response \f$ \Upsilon_Q \f$ on one cut of
  /// \f$ W \f$, evaluated at the declared sample points.
  ///
  /// @param action The joint action whose Ward current supplies the
  ///   restrictions, declared over a triangulation of \f$ W \f$.
  /// @param cobordism The interaction cobordism \f$ W \f$.
  /// @param cut A cooriented cut of \f$ W \f$.
  /// @param samples The points \f$ \lambda \f$ at which \f$ \Upsilon_Q \f$ and
  ///   its slope are evaluated. May be empty, and then only the poles and
  ///   residues are produced.
  /// @param cfg The declared parameters.
  /// @throws std::invalid_argument under the same conditions as `flux`.
  [[nodiscard]] static IntrinsicResponseRead intrinsicResponse(
      const JointAction &action,
      const observables::InteractionCobordism &cobordism,
      const observables::CoorientedCut &cut,
      const std::vector<std::complex<double>> &samples,
      const IntrinsicResponseConfig &cfg = {});
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_WARDFLUX_H
