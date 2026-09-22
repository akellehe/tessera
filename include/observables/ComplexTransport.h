// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_COMPLEXTRANSPORT_H
#define TESSERA_OBSERVABLES_COMPLEXTRANSPORT_H

// The complex fibre transport of matched Riesz frames: the general-linear map
// M_AB, the leakage certificate measured before restriction, the reversal by
// transposition, the Kato parallel transport of an isolated band, and the
// interferometric exchange character of two general-linear holonomies.
//
// ## Contents
//
//   • ComplexTransport   — the kernel. It forms the fibre map
//                          M_AB = Phi~_A^T T_AB Phi_B of a chain-level
//                          transfer T_AB between two matched bands, retains it
//                          as an element of GL(r, C), measures the leakage
//                          ||(I - P_A) T_AB P_B|| of the transfer before the
//                          restriction to the bands, reverses a link by
//                          transposition, composes links into a holonomy,
//                          carries an isolated band along a path of Riesz
//                          projectors by the Kato equation, and forms the
//                          exchange character of an exchange holonomy against
//                          a matched reference holonomy.
//   • GeneralLinearTransportRead
//                        — one transport A <- B: the unprojected map, its
//                          determinant, its singular data and conditioning,
//                          the leakage, the endpoint resolvent bounds and the
//                          endpoints' left and right frame residuals.
//   • KatoTransportRead  — the parallel transport of an isolated band along a
//                          path of Riesz projectors, with the intertwining
//                          residual that certifies it and the worst projector
//                          step that bounds it.
//   • ExchangeCharacterRead
//                        — chi_F = det(H_ex H_ref^-1), the complex number
//                          itself, with its modulus reported rather than
//                          required.
//
// ## What is retained, and what is never taken
//
// The retained observable is the complex matrix M_AB itself. No polar factor
// is extracted, no compact real form is imposed, no cube root of the
// determinant is selected on any link, no phase angle is read off a
// determinant, and no quantity is projected to unit modulus. Under
// independent frame changes at the two ends the map transforms exactly as
//
//     M_AB  |-->  g_A^-1 M_AB g_B,
//
// so a closed holonomy transforms by conjugation at its base point and its
// power traces, characteristic polynomial, determinant and conjugacy class are
// frame-free observables. That covariance is the reason no normalization is
// applied: a polar factor, a determinant root or a modulus is a choice of
// representative and destroys either the determinant transport or the frame
// law.
//
// ## Leakage
//
// The leakage certifies the approximation; it does not replace M_AB as the
// observable. It is measured on the transfer before the restriction to the
// bands,
//
//     leak_AB = ||(I - P_A) T_AB P_B||_2,
//
// the operator norm of the part of the transfer that starts inside the source
// band and lands outside the destination band. P_A and P_B are the two bands'
// Riesz projectors, which are oblique in the non-normal regime and orthogonal
// in the self-adjoint one. The quantity carries the scale of T_AB, so the
// relative leakage leak_AB / ||T_AB P_B||_2 is reported beside it and is the
// quantity a scale-free tolerance is applied to.
//
// ## Reversal
//
// A hopping-defined transfer reverses by transposition rather than by the
// conjugate adjoint: T_BA(U^-1) = T_AB(U)^T, and T_BA = T_AB^T at trivial
// connection. The transport assigned to an anti-cluster is the branch-free
// dual
//
//     M^v_AB = M_AB^-T,        det M^v_AB = (det M_AB)^-1,
//
// which follows from the reversal only under the further hypothesis that the
// two directions are groupoid inverses, T_BA = T_AB^-1. That hypothesis is not
// checked here, because it is a property of the gluing and not of the matrix;
// `dualTransport` therefore states it rather than asserting it, and
// `reversedTransfer` implements the transposition alone.
//
// ## Kato transport
//
// For a differentiable family of Riesz projectors P(t) of an isolated band,
// the Kato equation
//
//     K' = [P', P] K,        K(0) = I,
//
// transports the band along the family: its solution satisfies
// K(t) P(0) = P(t) K(t) exactly, so it carries the band's range onto the
// band's range at every parameter without ever leaving the isolated bundle.
// A sampled path is carried step by step; `KatoScheme` names which step is
// used and what each one is exact for.
//
// Everything here is a pure function of caller-supplied data: no solver call,
// no Spacetime mutation, and nothing enters the emergence objective.

#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cobordism/Certificate.h"
#include "observables/SheetedColor.h"
#include "observables/SpectralFiber.h"

namespace tessera::observables {

/// Threshold configuration of the complex-transport reads. A threshold selects
/// which reads are certified, never which value is reported: a failed
/// threshold yields an uncertified read carrying the same numbers.
struct ComplexTransportConfig {
  /// Relative singular-value cut deciding the numerical rank of a map: the
  /// singular value sigma_i counts toward the rank when
  /// sigma_i > rankTolerance * sigma_max.
  double rankTolerance = 1e-9;
  /// Cap on the relative leakage leak_AB / ||T_AB P_B||_2 below which the
  /// restriction of the transfer to the two bands is certified. The absolute
  /// leakage carries the scale of the transfer, so the tolerance is applied to
  /// the relative quantity.
  double leakageTolerance = 1e-9;
  /// Cap on the conditioning that a certified read may carry: the map's own
  /// sigma_max/sigma_min and each endpoint band's projector norm ||P||_2 must
  /// both stay below it.
  double conditionNumberCap = 1e8;
  /// Absolute floor on each endpoint band's isolation, the distance in the
  /// complex plane from the band to the nearest eigenvalue outside it. Zero
  /// means rely on the bands' own certification rather than impose a second
  /// floor.
  double isolationFloor = 0.0;
  /// Require both endpoint bands to carry an accepted band certificate. A band
  /// whose isolating gap has closed is not a band, and no transport is
  /// certified between one.
  bool requireCertifiedFibers = true;
  /// Tolerance the emitted certificates hold against for the residuals that
  /// are exact in exact arithmetic: the Kato intertwining residual and the
  /// agreement between the two routes to the exchange character.
  double certificateTolerance = 1e-10;
};

/// # GeneralLinearTransportRead
///
/// One complex fibre transport A <- B. The map is reported unprojected,
/// together with everything the whitepaper asks to be reported beside it: the
/// determinant, the condition number, the leakage, the endpoint resolvent
/// bounds and the endpoints' left and right frame residuals. A quantity that
/// was not measured is a quiet NaN, never zero.
struct GeneralLinearTransportRead {
  /// Form degree of the two bands.
  int degree = 0;
  /// The common frame rank r of the two bands. When the ranks disagree the
  /// read is rejected and this holds the destination rank.
  int rank = 0;
  /// M_AB = Phi~_A^T T_AB Phi_B, the retained element of GL(r, C). Phi~_A is
  /// the destination band's transpose dual frame, normalized by
  /// Phi~_A^T Phi_A = I, and Phi_B is the source band's right frame. No polar
  /// factor, compact real form or determinant root is applied to it.
  Eigen::MatrixXcd map{};
  /// det M_AB, the determinant-line datum. It is an element of C*, reported as
  /// the complex number it is: neither its modulus nor its argument is taken.
  std::complex<double> determinant{0.0, 0.0};
  /// Singular values of `map`, descending.
  std::vector<double> singularValues{};
  /// Numerical rank of `map` at `ComplexTransportConfig::rankTolerance`.
  int numericalRank = 0;
  /// sigma_max / sigma_min of `map`; infinity when the map is singular, which
  /// is the one analytic failure mode of the transport.
  double conditionNumber = std::numeric_limits<double>::quiet_NaN();
  /// The smallest singular value of `map`: the distance to the rank-dropping
  /// configuration that a determinant winding encircles.
  double minSingularValue = std::numeric_limits<double>::quiet_NaN();
  /// leak_AB = ||(I - P_A) T_AB P_B||_2, measured on the transfer before the
  /// restriction to the bands.
  double leakage = std::numeric_limits<double>::quiet_NaN();
  /// leak_AB / ||T_AB P_B||_2, the scale-free form of the same quantity and
  /// the one `ComplexTransportConfig::leakageTolerance` caps. Zero when the
  /// restricted transfer itself vanishes, where there is nothing to leak.
  double relativeLeakage = std::numeric_limits<double>::quiet_NaN();
  /// The destination band's isolation: the distance in the complex plane from
  /// the band to the nearest eigenvalue outside it.
  double toIsolation = std::numeric_limits<double>::quiet_NaN();
  /// The source band's isolation, in the same sense.
  double fromIsolation = std::numeric_limits<double>::quiet_NaN();
  /// The destination band's resolvent bound ||P_A||_2 / g_A, with g_A its
  /// isolation: the factor that bounds the norm of the resolvent on a contour
  /// running at distance g_A from the band, and hence the sensitivity of the
  /// Riesz projector that defines the band. Infinity when the isolation is
  /// zero, where the band is not isolated and no bound exists.
  double toResolventBound = std::numeric_limits<double>::quiet_NaN();
  /// The source band's resolvent bound, in the same sense.
  double fromResolventBound = std::numeric_limits<double>::quiet_NaN();
  /// The destination band's right-frame residual: the eigenvalue residual
  /// ||L Phi - Phi Lambda||_F / ||L||_F of the band that produced it.
  double toRightFrameResidual = std::numeric_limits<double>::quiet_NaN();
  /// The destination band's left-frame residual
  /// ||L^dagger Y - Y Lambda^dagger||_F / ||L||_F, equal to the right-frame
  /// residual on the self-adjoint path where the two frames coincide.
  double toLeftFrameResidual = std::numeric_limits<double>::quiet_NaN();
  /// The source band's right-frame residual.
  double fromRightFrameResidual = std::numeric_limits<double>::quiet_NaN();
  /// The source band's left-frame residual.
  double fromLeftFrameResidual = std::numeric_limits<double>::quiet_NaN();
  /// The destination band's projector norm ||P_A||_2: one for an orthogonal
  /// projector and larger the more oblique the band.
  double toProjectorNorm = std::numeric_limits<double>::quiet_NaN();
  /// The source band's projector norm.
  double fromProjectorNorm = std::numeric_limits<double>::quiet_NaN();
  /// The metric regime of the pair, the worse of the two endpoints'.
  cobordism::CertificateRegime regime =
      cobordism::CertificateRegime::NonNormal;
  /// Whether `map` reached full numerical rank, so that it really is an
  /// element of GL(r, C) and its inverse and dual exist.
  bool invertible = false;
  /// Whether every gate passed: matched ranks, certified isolated endpoint
  /// bands, conditioning below the cap, full rank, and relative leakage below
  /// the tolerance.
  bool accepted = false;
  /// The first failed gate, named; empty when accepted.
  std::string rejectionReason{};
  /// The graded record. An accepted transport is CertifiedNumerical on the
  /// BandWindow domain in the pair's regime, grading the relative leakage
  /// against `ComplexTransportConfig::leakageTolerance` with the map's
  /// condition number as the conditioning. A rejected transport carries the
  /// never-holding HeuristicDiscovery grade and keeps every number above.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary: rank, determinant, conditioning,
  /// leakage and whether the gates passed.
  [[nodiscard]] std::string describe() const;
};

/// Which step carries the band from one sampled Riesz projector to the next,
/// and what that step is exact for. Naming the scheme is required rather than
/// inferred, so that no approximation is ever applied silently.
enum class KatoScheme {
  /// The direct rotation
  /// W = (I - (P1 - P0)^2)^-1/2 (P1 P0 + (I - P1)(I - P0)).
  /// For orthogonal projectors this is the exact solution of the Kato equation
  /// along the geodesic family joining P0 to P1 in the Grassmannian, and it is
  /// unitary; the intertwining W P0 = P1 W is exact to rounding. It needs
  /// I - (P1 - P0)^2 to have no eigenvalue on the negative real axis, which
  /// holds whenever the two projectors are closer than one in operator norm.
  DirectRotation,
  /// The unnormalized intertwiner R = P1 P0 + (I - P1)(I - P0). The identity
  /// R P0 = P1 R is exact for any pair of idempotents, orthogonal or oblique,
  /// so this is the step that survives the oblique Riesz projectors of a
  /// non-normal operator. It is not normalized, so it carries a scale on the
  /// band that the direct rotation removes.
  Intertwiner,
  /// The exponential of the Kato generator of the step,
  /// exp([P1, P0]). The commutator [P1, P0] is the exact integral of the Kato
  /// generator [P', P] over a step of the geodesic family to second order, so
  /// the intertwining residual of this step is third order in ||P1 - P0|| and
  /// is reported rather than assumed away. Offered because it is the Kato
  /// equation's own exponential step; it is never selected implicitly.
  ExponentialGenerator,
};

/// # KatoTransportRead
///
/// The parallel transport of an isolated band along a sampled path of Riesz
/// projectors. The reported transport is the ordered product of the steps,
/// and the certificate grades the one premise the construction has: that each
/// step really intertwines the two projectors it joins.
struct KatoTransportRead {
  /// K = K_{n-1} ... K_0, the composed transport from the first projector of
  /// the path to the last. Empty when the read was structurally invalidated.
  Eigen::MatrixXcd transport{};
  /// The per-step transports K_t in path order, so that a caller may inspect
  /// or recompose them.
  std::vector<Eigen::MatrixXcd> stepTransports{};
  /// The scheme the steps were built with.
  KatoScheme scheme = KatoScheme::DirectRotation;
  /// The number of steps, one fewer than the number of projectors.
  std::size_t steps = 0;
  /// The common dimension of the projectors.
  std::size_t dimension = 0;
  /// The rank of the transported band, the trace of the first projector
  /// rounded to the nearest integer. It is constant along an isolated family,
  /// and `rankDefect` measures how far the path kept it so.
  std::size_t rank = 0;
  /// max over the path of |tr P_t - rank|: the departure of the band's rank
  /// from the integer it must hold at, which a path that loses isolation
  /// shows.
  double rankDefect = std::numeric_limits<double>::quiet_NaN();
  /// max over the path of ||P_t^2 - P_t||_2: how far the supplied projectors
  /// are from idempotent, the premise every step rests on.
  double idempotencyResidual = std::numeric_limits<double>::quiet_NaN();
  /// max over the steps of ||K_t P_t - P_{t+1} K_t||_2: the intertwining
  /// residual, zero to rounding for the two exact schemes and third order in
  /// the step for the exponential one.
  double intertwiningResidual = std::numeric_limits<double>::quiet_NaN();
  /// ||K P_0 - P_n K||_2 of the composed transport against the two ends of the
  /// path.
  double composedIntertwiningResidual =
      std::numeric_limits<double>::quiet_NaN();
  /// max over the steps of ||P_{t+1} - P_t||_2: the coarseness of the sampling,
  /// which bounds the exponential scheme's residual and which must stay below
  /// one for the direct rotation to exist.
  double maxProjectorStep = std::numeric_limits<double>::quiet_NaN();
  /// Whether every step was formed. False when the direct rotation's square
  /// root did not exist or a step was singular, in which case `transport` is
  /// empty and the reason is named.
  bool complete = false;
  /// Why the transport was not formed, when it was not; empty otherwise.
  std::string invalidReason{};
  /// The graded record: CertifiedNumerical on the BandWindow domain, grading
  /// the intertwining residual against
  /// `ComplexTransportConfig::certificateTolerance`, with the worst projector
  /// step as the conditioning. An incomplete transport carries the
  /// never-holding HeuristicDiscovery grade.
  cobordism::Certificate certificate{};
};

/// # ExchangeCharacterRead
///
/// The interferometric exchange character of a general-linear exchange
/// holonomy against the holonomy of a non-exchanging reference path of the
/// same geometric footprint, timing, contours and local frame convention:
///
///     chi_F = det(H_ex H_ref^-1)  in  C*.
///
/// The complex number is the whole result. Its modulus is reported because a
/// reader wants to see it, not because anything is required of it, and no
/// phase angle, component sign or unit-modulus projection is taken anywhere.
struct ExchangeCharacterRead {
  /// chi_F = det(H_ex H_ref^-1), computed as the determinant of the composed
  /// matrix, exactly as written.
  std::complex<double> character{std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN()};
  /// det H_ex / det H_ref, the same number by multiplicativity of the
  /// determinant, computed independently as a cross-check of the composition.
  std::complex<double> determinantRatio{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// |character - determinantRatio|, the agreement of the two routes. It is
  /// zero in exact arithmetic and is what the certificate's residual grades
  /// beside the leakage.
  double routeAgreementResidual = std::numeric_limits<double>::quiet_NaN();
  /// det H_ex, the exchange holonomy's determinant on its own.
  std::complex<double> exchangeDeterminant{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// det H_ref, the reference holonomy's determinant on its own.
  std::complex<double> referenceDeterminant{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// |chi_F|. Reported, never required: a modulus away from one is a fact
  /// about the realized motion and is not corrected by projection.
  double modulus = std::numeric_limits<double>::quiet_NaN();
  /// |chi_F + 1|, the distance to the value a single exchange of an odd-rank
  /// pair predicts.
  double distanceToMinusOne = std::numeric_limits<double>::quiet_NaN();
  /// |chi_F - 1|, the distance to the value a double exchange predicts.
  double distanceToPlusOne = std::numeric_limits<double>::quiet_NaN();
  /// The common rank r of the two holonomies.
  int rank = 0;
  /// sigma_max/sigma_min of the reference holonomy, whose inverse the
  /// character takes; infinity when the reference is singular.
  double referenceConditionNumber = std::numeric_limits<double>::quiet_NaN();
  /// The worst leakage encountered along the two paths, as the caller
  /// measured it. This, and the isolation of the bands, is what the exchange
  /// experiment actually tests; the sign itself is algebraic.
  double pathLeakage = std::numeric_limits<double>::quiet_NaN();
  /// Whether the reference holonomy could be inverted at all. A singular
  /// reference leaves the character unmeasured rather than large.
  bool referenceInvertible = false;
  /// The graded record: CertifiedNumerical on the BandWindow domain grading
  /// the larger of the supplied path leakage and the route-agreement residual
  /// against `ComplexTransportConfig::leakageTolerance`, with the reference
  /// holonomy's condition number as the conditioning. The modulus of the
  /// character enters no conjunct of this certificate.
  cobordism::Certificate certificate{};
};

/// # ComplexTransport
///
/// The complex fibre transport of matched Riesz frames, its leakage
/// certificate, its reversal, its Kato parallel transport and the exchange
/// character of two general-linear holonomies. See the file banner for the
/// identities and for what is never taken.
///
/// Every member is static: each is a pure function of the declared transfer,
/// the supplied bands and caller-supplied frames. The class consumes bands,
/// never re-extracts them, mutates nothing, and none of its outputs enters any
/// emergence objective.
class ComplexTransport {
 public:
  /// Double-precision complex scalar of every matrix and frame entry.
  using Complex = std::complex<double>;

  ComplexTransport() = delete;  // static kernel — no instances.

  // ── the fibre map ───────────────────────────────────────────────────────

  /// The fibre map M_AB = Phi~_A^T T_AB Phi_B of a chain-level transfer.
  ///
  /// `dualFrameTo` is the destination band's transpose dual frame Phi~_A,
  /// normalized by Phi~_A^T Phi_A = I; `transfer` is T_AB, with rows the
  /// destination's cells and columns the source's cells; `rightFrameFrom` is
  /// the source band's right frame Phi_B. The pairing is bilinear — the dual
  /// frame enters by its transpose and never by its conjugate — which is what
  /// makes the result transform by g_A^-1 M g_B rather than by a unitary law.
  /// @throws std::invalid_argument on a shape mismatch between the three
  ///         factors.
  [[nodiscard]] static Eigen::MatrixXcd fiberMap(
      const Eigen::MatrixXcd& dualFrameTo, const Eigen::MatrixXcd& transfer,
      const Eigen::MatrixXcd& rightFrameFrom);

  /// The leakage ||(I - P_A) T_AB P_B||_2 of a transfer against the two bands'
  /// Riesz projectors, measured before the restriction to the bands.
  ///
  /// `projectorTo` is P_A on the destination's cells, `projectorFrom` is P_B on
  /// the source's cells, and `transfer` maps the source's cells to the
  /// destination's.
  /// @throws std::invalid_argument on a shape mismatch, or when either
  ///         projector is not square.
  [[nodiscard]] static double leakage(const Eigen::MatrixXcd& projectorTo,
                                      const Eigen::MatrixXcd& transfer,
                                      const Eigen::MatrixXcd& projectorFrom);

  /// The complete transport A <- B of a transfer between two bands: the
  /// unprojected map, its determinant and singular data, the leakage measured
  /// before restriction, the endpoint resolvent bounds, the endpoints' left
  /// and right frame residuals, the gates and the certificate.
  ///
  /// `to` and `from` are the destination and source bands, `transfer` has the
  /// destination's cells as its rows and the source's cells as its columns.
  /// @throws std::invalid_argument when the two bands sit at different
  ///         degrees, or when the transfer's shape does not match the two
  ///         bands' cell counts.
  [[nodiscard]] static GeneralLinearTransportRead transport(
      const SpectralFiber& to, const SpectralFiber& from,
      const Eigen::MatrixXcd& transfer, const ComplexTransportConfig& cfg = {});

  /// The frame law M_AB |--> g_A^-1 M_AB g_B under independent frame changes
  /// at the two ends. It delegates to `SheetAttachment::frameChanged`, which
  /// is the same identity written for the sheet factor; the law is one
  /// statement and lives in one place.
  /// @throws std::invalid_argument on a shape mismatch or a singular g_A.
  [[nodiscard]] static Eigen::MatrixXcd frameChanged(
      const Eigen::MatrixXcd& map, const Eigen::MatrixXcd& frameTo,
      const Eigen::MatrixXcd& frameFrom);

  /// The reversed chain transfer T_BA = T_AB^T.
  ///
  /// This is the reversal a hopping-defined transfer obeys at trivial
  /// connection; with a connection the same identity reads
  /// T_BA(U^-1) = T_AB(U)^T, so the caller supplies the transfer evaluated at
  /// the reversed connection and this function transposes it. It is the
  /// transpose and not the conjugate adjoint: a complex squared length is not
  /// conjugated by traversing its edge backwards.
  [[nodiscard]] static Eigen::MatrixXcd reversedTransfer(
      const Eigen::MatrixXcd& transfer);

  /// The branch-free dual transport M^v_AB = M_AB^-T assigned to an
  /// anti-cluster, whose determinant is (det M_AB)^-1 exactly.
  ///
  /// It follows from the transposition reversal only under the further
  /// hypothesis that the two directions of the transfer are groupoid inverses,
  /// T_BA = T_AB^-1. That hypothesis is a property of the gluing and cannot be
  /// read off the matrix, so it is stated here and not checked: a caller whose
  /// gluing does not satisfy it obtains the sign of the determinant winding
  /// from its closure convention instead.
  /// @throws std::invalid_argument when `map` is not square, is empty, or is
  ///         numerically singular, in which case no dual exists.
  [[nodiscard]] static Eigen::MatrixXcd dualTransport(
      const Eigen::MatrixXcd& map);

  /// The transport along a declared path, applied right to left:
  /// `compose({M_1, ..., M_n})` returns M_n ... M_1, the transport of the path
  /// that traverses M_1 first. It delegates to `SheetAttachment::compose`, the
  /// same ordered general-linear product.
  /// @throws std::invalid_argument on an empty path with a zero rank, a
  ///         non-square factor, or a shape mismatch between consecutive
  ///         factors.
  [[nodiscard]] static Eigen::MatrixXcd compose(
      const std::vector<Eigen::MatrixXcd>& path, std::size_t rank = 0);

  /// The closed holonomy of a link sequence and its conjugacy invariants —
  /// the ordered product, its power traces, the characteristic polynomial they
  /// determine and its determinant. It delegates to
  /// `SheetAttachment::holonomy`; the links are supplied in traversal order,
  /// so the first is applied first.
  /// @throws std::invalid_argument on an empty sequence or a shape mismatch.
  [[nodiscard]] static HolonomyInvariants holonomy(
      const std::vector<Eigen::MatrixXcd>& links);

  // ── Kato transport ──────────────────────────────────────────────────────

  /// The right-hand side [P', P] of the Kato equation for an explicit
  /// projector rate: `projectorRate` is P' and `projector` is P.
  /// @throws std::invalid_argument when the two are not square and of the same
  ///         size.
  [[nodiscard]] static Eigen::MatrixXcd katoGenerator(
      const Eigen::MatrixXcd& projectorRate, const Eigen::MatrixXcd& projector);

  /// One Kato step from the projector `from` to the projector `to` under the
  /// named scheme. Each scheme and what it is exact for is documented on
  /// `KatoScheme`.
  /// @throws std::invalid_argument when the two projectors are not square and
  ///         of the same size, or when the direct rotation's square root does
  ///         not exist because the two projectors are not closer than one in
  ///         operator norm.
  [[nodiscard]] static Eigen::MatrixXcd katoStep(
      const Eigen::MatrixXcd& from, const Eigen::MatrixXcd& to,
      KatoScheme scheme = KatoScheme::DirectRotation);

  /// The Kato parallel transport of an isolated band along a sampled path of
  /// Riesz projectors, with the residuals that certify it.
  ///
  /// `projectors` are P_0, ..., P_n in path order, all square and of the same
  /// size; a closed path repeats its first projector as its last, which makes
  /// the transport the band's holonomy around the loop. A step that could not
  /// be formed leaves the read incomplete with the reason named rather than
  /// returning a transport that does not intertwine.
  /// @throws std::invalid_argument for fewer than two projectors, or when the
  ///         projectors are not all square and of one size.
  [[nodiscard]] static KatoTransportRead katoTransport(
      const std::vector<Eigen::MatrixXcd>& projectors,
      KatoScheme scheme = KatoScheme::DirectRotation,
      const ComplexTransportConfig& cfg = {});

  /// The Kato transport of the path of bands `loop`, whose Riesz projectors
  /// are read from the bands themselves. Every band on the path must carry the
  /// same cell list, in the same order, because a single transport matrix
  /// exists only on one cell space; a path over changing supports is
  /// transported link by link with `transport` instead.
  /// @throws std::invalid_argument for fewer than two bands, or when two bands
  ///         on the path carry different cells.
  [[nodiscard]] static KatoTransportRead katoTransportOnFibers(
      const std::vector<SpectralFiber>& loop,
      KatoScheme scheme = KatoScheme::DirectRotation,
      const ComplexTransportConfig& cfg = {});

  /// The r x r matrix of a Kato transport in the two ends' band frames:
  /// k = Phi~_end^T K Phi_start. This is the same bilinear pairing as
  /// `fiberMap`, applied to the transport instead of to a chain transfer, and
  /// it is the general-linear link a Kato-transported band contributes to a
  /// holonomy.
  /// @throws std::invalid_argument on a shape mismatch between the three
  ///         factors.
  [[nodiscard]] static Eigen::MatrixXcd bandTransport(
      const Eigen::MatrixXcd& transport, const Eigen::MatrixXcd& dualFrameEnd,
      const Eigen::MatrixXcd& rightFrameStart);

  // ── the exchange character ──────────────────────────────────────────────

  /// chi_F = det(H_ex H_ref^-1) of an exchange holonomy against the holonomy
  /// of a matched non-exchanging reference path.
  ///
  /// Both holonomies are general-linear matrices of the same rank, unprojected
  /// and with no determinant root taken. `pathLeakage` is the worst leakage
  /// the caller measured along the two paths — the quantity the experiment
  /// actually tests, together with the isolation of the bands — and it enters
  /// the certificate; the modulus of the character does not.
  /// @throws std::invalid_argument when either holonomy is empty or not
  ///         square, or when the two have different sizes.
  [[nodiscard]] static ExchangeCharacterRead exchangeCharacter(
      const Eigen::MatrixXcd& exchangeHolonomy,
      const Eigen::MatrixXcd& referenceHolonomy, double pathLeakage,
      const ComplexTransportConfig& cfg = {});

  /// `exchangeCharacter` of the two paths' composed holonomies: the exchange
  /// path and the reference path are supplied as their ordered link sequences,
  /// each traversed first link first, and composed by `compose`.
  /// @throws std::invalid_argument on an empty path, a non-square link, or a
  ///         shape mismatch anywhere.
  [[nodiscard]] static ExchangeCharacterRead exchangeCharacterOfPaths(
      const std::vector<Eigen::MatrixXcd>& exchangePath,
      const std::vector<Eigen::MatrixXcd>& referencePath, double pathLeakage,
      const ComplexTransportConfig& cfg = {});
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_COMPLEXTRANSPORT_H
