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

namespace tessera::cobordism {

/// # CooorientedCut
///
/// A cooriented separating cut \f$ \Sigma \f$ of Section 13.1 of the
/// whitepaper, declared by the vertices that lie on its incoming side.
///
/// The cobordism supplies the cut: with the oriented boundary convention
/// \f$ \partial W = \partial_{\rm in}W \sqcup \partial_{\rm out}W \f$, a
/// separating slice is a cooriented closed codimension-one simplicial cut
/// separating \f$ \partial_{\rm in}W \f$ from \f$ \partial_{\rm out}W \f$. On
/// the 1-skeleton such a cut is fixed by the vertex set it leaves behind, so
/// that is what is declared here: `incomingSide` is the set of vertex
/// identifiers on the \f$ \partial_{\rm in}W \f$ side, and the cut is the set
/// of edges with exactly one endpoint in it.
///
/// The coorientation is the one the cobordism induces and is carried on each
/// cut edge as \f$ +1 \f$ when the edge leaves the incoming side (its lower
/// vertex is inside and its upper vertex is outside) and \f$ -1 \f$ when it
/// enters. Reversing the global cobordism orientation — declaring the
/// complementary vertex set as the incoming side — reverses every cut edge's
/// coorientation together and so reverses the flux, which is the sign rule the
/// whitepaper states.
///
/// No Lorentzian distance, real projection or level-set ordering enters: the
/// cut is a combinatorial object on the 1-skeleton and nothing here reads a
/// vertex time or a temporal function.
struct CooorientedCut {
  /// Vertex identifiers on the incoming side of the cut. Order is irrelevant
  /// and repeats are ignored.
  std::vector<std::uint64_t> incomingSide{};
  /// The caller's label for the cut, echoed on the read so several cuts of one
  /// homology class can be told apart in a report.
  std::string label{};
};

/// # WardFluxConfig
///
/// Every threshold of the Ward-flux read, echoed on each read so a stored
/// result carries the configuration that produced it.
struct WardFluxConfig {
  /// \f$ |(\partial j)_x| \f$ at or below this counts as a vanishing
  /// divergence when the bulk conservation certificate is judged. It is an
  /// absolute number and is compared against the divergence measured at the
  /// vertices strictly inside the incoming side.
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
};

/// # WardFluxRead
///
/// The flux \f$ \varphi_j(\Sigma)=\langle j,\Sigma\rangle \in \mathbb{C} \f$ of
/// the complex Ward current through one cooriented cut, with every certificate
/// the whitepaper attaches to it.
struct WardFluxRead {
  /// The caller's label for the cut this read belongs to.
  std::string label{};
  /// The cut's edges as canonical degree-one cell indices, ascending.
  std::vector<int> cutCells{};
  /// The coorientation \f$ c_e \in \{+1,-1\} \f$ of each cut edge, parallel to
  /// `cutCells`: \f$ +1 \f$ when the edge leaves the incoming side.
  std::vector<int> coorientation{};
  /// The Ward current \f$ j_e \f$ on each cut edge, on the canonical
  /// ascending-vertex orientation and parallel to `cutCells`.
  std::vector<std::complex<double>> cutCurrent{};

  /// \f$ \varphi_j(\Sigma)=\sum_{e\in\Sigma} c_e\,j_e \f$, the flux. Complex,
  /// and never projected onto a real part.
  std::complex<double> flux{0.0, 0.0};

  /// \f$ \sum_{x\in\Omega}(\partial j)_x \f$ over the incoming side
  /// \f$ \Omega \f$. With the chain complex's boundary convention
  /// \f$ \partial[x<y]=[y]-[x] \f$ this is exactly \f$ -\varphi_j(\Sigma) \f$:
  /// the discrete divergence theorem, which holds identically and is reported
  /// so that the identity is measured rather than assumed.
  std::complex<double> enclosedDivergence{0.0, 0.0};
  /// \f$ |\varphi_j(\Sigma)+\sum_{x\in\Omega}(\partial j)_x| \f$, the residual
  /// of that identity. Zero to rounding on every read.
  double divergenceTheoremResidual = std::numeric_limits<double>::quiet_NaN();

  /// \f$ \max_x |(\partial j)_x| \f$ over the vertices strictly inside the
  /// incoming side, i.e. those the cut does not touch. This is the discrete
  /// Ward identity \f$ \partial j=0 \f$ in the bulk, measured. NaN when the
  /// incoming side has no strictly interior vertex.
  double bulkDivergenceMax = std::numeric_limits<double>::quiet_NaN();
  /// How many vertices `bulkDivergenceMax` was taken over.
  std::size_t bulkVertices = 0;
  /// The vertex identifier at which `bulkDivergenceMax` was attained; empty
  /// when there is no strictly interior vertex. A source in the bulk is named
  /// by its vertex rather than reported as an anonymous norm.
  std::optional<std::uint64_t> bulkDivergenceVertex{};

  /// \f$ \sum_{\sigma\subset\Omega}\Gamma_{\sigma\sigma} \f$, the fermion
  /// number the declared covariance places on the carrier cells whose vertices
  /// all lie on the incoming side. This is the number Section 13.4 states the
  /// flux equals, computed independently of the current so that the statement
  /// is a measurement and not a definition. Empty when the action declares no
  /// covariance.
  std::optional<std::complex<double>> enclosedFermionNumber{};
  /// How many carrier cells lie wholly on the incoming side.
  std::size_t enclosedCells = 0;
  /// \f$ |\varphi_j(\Sigma)-\sum_{\sigma\subset\Omega}\Gamma_{\sigma\sigma}|
  /// \f$, the defect of the identity "the flux equals the enclosed fermion
  /// number".
  /// NaN when no covariance is declared.
  double fermionNumberResidual = std::numeric_limits<double>::quiet_NaN();

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

  /// Whether the declared vertex set is a separating cut: both sides nonempty
  /// and each side connected in the 1-skeleton, so the cut is one closed
  /// codimension-one slice rather than several.
  bool separating = false;
  /// Named failures: "empty-incoming-side", "empty-outgoing-side",
  /// "disconnected-incoming-side", "disconnected-outgoing-side",
  /// "empty-cut", "bulk-source", "nonintegral-flux", "complex-flux".
  std::vector<std::string> failedCertificates{};
};

/// # WardHomologyRead
///
/// Several cuts of one homology class read together: the whitepaper's
/// requirement that different homologous cuts give the same current flux
/// whenever no source lies in the slab between them.
struct WardHomologyRead {
  /// One read per declared cut, in the order the cuts were supplied.
  std::vector<WardFluxRead> cuts{};
  /// \f$ \max_{i,j}|\varphi_j(\Sigma_i)-\varphi_j(\Sigma_j)| \f$ over the
  /// declared cuts. NaN for fewer than two cuts.
  double maxFluxDeviation = std::numeric_limits<double>::quiet_NaN();
  /// \f$ \max_{i<j}\sum_{x\in\Omega_i\triangle\Omega_j}|(\partial j)_x| \f$:
  /// the total divergence carried by the slabs between the cuts, which is the
  /// source content the invariance statement excludes. A nonzero deviation
  /// with a nonzero slab divergence is a source, not a broken identity.
  double maxSlabDivergence = std::numeric_limits<double>::quiet_NaN();
  /// Whether every pair agreed to `divergenceTolerance`.
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
  /// The cut's label.
  std::string label{};
  /// The cut's edges as canonical degree-one cell indices, ascending. These
  /// are the slice's coordinates: \f$ L_\Sigma \f$, \f$ \rho_R \f$ and
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
  /// degree-one metric Hodge operator \f$ h_1(z,U) \f$ on the cut's cells,
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
/// The flux of the complex Ward current through a cooriented cut (Section
/// 13.4) and the intrinsic spectral response it carries (Section 13.5).
///
/// The current itself is the link stationarity vector of the joint action,
/// \f$ j_{xy}=U_{xy}\,\partial S/\partial U_{xy} \f$, which `JointAction`
/// supplies under both of its names. Nothing here re-derives it, and nothing
/// here supplies a field of its own: the audited substitutes for this
/// observable summed a caller-supplied field-strength cochain over a closed
/// star, which is a different object with no Ward identity behind it.
///
/// ## The flux
///
/// For a cut \f$ \Sigma \f$ with incoming side \f$ \Omega \f$,
/// \f[
///   \varphi_j(\Sigma)=\langle j,\Sigma\rangle=\sum_{e\in\Sigma}c_e\,j_e ,
/// \f]
/// with \f$ c_e=+1 \f$ on an edge leaving \f$ \Omega \f$ and \f$ -1 \f$ on one
/// entering it. Because the chain complex's boundary convention is
/// \f$ \partial[x<y]=[y]-[x] \f$, the discrete divergence theorem reads
/// \f$ \varphi_j(\Sigma)=-\sum_{x\in\Omega}(\partial j)_x \f$, and both sides
/// are reported.
///
/// The Ward identity \f$ \partial j=0 \f$ holds in the bulk: every
/// gauge-invariant term of the action contributes nothing to the divergence at
/// all, and the matter term contributes
/// \f$ w_M\sum_{\sigma:\,v_0(\sigma)=x}[h,\Gamma]_{\sigma\sigma} \f$, which
/// vanishes exactly when the declared covariance commutes with the carrier
/// operator — which is what a spectral projector of \f$ h \f$ does. The bulk
/// divergence is therefore measured on every read and named as a source when
/// it does not vanish, rather than being asserted to be zero.
///
/// Section 13.4 states that the flux equals the fermion number enclosed by the
/// cut, \f$ N_q \f$, three times baryon number. Both numbers are reported:
/// the flux from the current, and the enclosed fermion number
/// \f$ \sum_{\sigma\subset\Omega}\Gamma_{\sigma\sigma} \f$ from the covariance,
/// together with their difference. The equality is a measurement of this
/// framework, not a definition inside it.
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
/// restrictions of the current to the cut and \f$ L_\Sigma \f$ the slice
/// operator. Its poles are the eigenvalues of \f$ L_\Sigma \f$, its residues
/// are taken through Riesz projectors so a degenerate band is handled whole,
/// and its slope in \f$ \lambda \f$ is the analytic second power of the
/// resolvent rather than a finite difference.
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
/// Read-only: it reads an action and a declared cut, never calls a solver,
/// never writes the geometry, and never enters an emergence objective. An
/// unmeasured value is NaN or an empty optional with the reason named, never
/// zero.
class WardFlux {
 public:
  /// The flux of the action's Ward current through one cut.
  ///
  /// @param action The joint action whose link stationarity is the current.
  /// @param cut The cooriented cut, declared by its incoming side.
  /// @param cfg The declared thresholds.
  /// @throws std::invalid_argument when the action carries no spacetime.
  [[nodiscard]] static WardFluxRead flux(const JointAction &action,
                                         const CooorientedCut &cut,
                                         const WardFluxConfig &cfg = {});

  /// Several cuts read together, with the pairwise flux deviation and the
  /// divergence carried by the slabs between them.
  [[nodiscard]] static WardHomologyRead homologousFluxes(
      const JointAction &action, const std::vector<CooorientedCut> &cuts,
      const WardFluxConfig &cfg = {});

  /// The coherent background removal
  /// \f$ \Delta\varphi=\varphi_{\rm state}-\varphi_{\rm matched} \f$ of two
  /// flux reads: the complex difference of the fluxes and of the enclosed
  /// fermion numbers, with no modulus taken on either side. The returned read
  /// carries the state read's cut and certificates and the differenced
  /// numbers.
  /// @throws std::invalid_argument when the two reads were taken on different
  ///   cut cells, since a difference between different cuts is not a
  ///   background removal.
  [[nodiscard]] static WardFluxRead difference(const WardFluxRead &state,
                                               const WardFluxRead &matched,
                                               const WardFluxConfig &cfg = {});

  /// The intrinsic spectral response \f$ \Upsilon_Q \f$ on one cut, evaluated
  /// at the declared sample points.
  ///
  /// @param action The joint action whose Ward current supplies the
  ///   restrictions.
  /// @param cut The cooriented cut.
  /// @param samples The points \f$ \lambda \f$ at which \f$ \Upsilon_Q \f$ and
  ///   its slope are evaluated. May be empty, and then only the poles and
  ///   residues are produced.
  /// @param cfg The declared parameters.
  /// @throws std::invalid_argument when the action carries no spacetime.
  [[nodiscard]] static IntrinsicResponseRead intrinsicResponse(
      const JointAction &action, const CooorientedCut &cut,
      const std::vector<std::complex<double>> &samples,
      const IntrinsicResponseConfig &cfg = {});
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_WARDFLUX_H
